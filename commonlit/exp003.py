# 特徴量も合わせて学習する
from typing import List
import numpy as np 
import pandas as pd 
import warnings
import logging
import os
import gc
import random
import pickle
import shutil
import subprocess
import json
import datetime
from pickle import dump, load
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from datasets import Dataset,load_dataset, load_from_disk
from transformers import TrainingArguments, Trainer
from datasets import load_metric, disable_progress_bar
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, GroupKFold
from oauth2client.service_account import ServiceAccountCredentials
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from collections import Counter
import spacy
import gensim
import re
from spellchecker import SpellChecker
import lightgbm as lgb

class CFG:
    model_name: str ="debertav3base"
    learning_rate: float =1.5e-5
    weight_decay: float =0.03
    hidden_dropout_prob: float =0.007
    attention_probs_dropout_prob: float =0.007
    num_train_epochs: int =5
    n_splits: int =4
    batch_size: int =12
    random_seed: int =42
    save_steps: int =50
    max_length: int =512
    n_freeze: int=4
    mean_pooling: bool=False

class RCFG:
    run_name: str = 'run'
    debug: bool =True
    debug_size: int =10
    train: bool = True
    predict: bool = True
    commit_hash: str =""
    base_model_dir: str = "/kaggle/input/debertav3base"
    output_path: str = ""
    model_dir: str = "." # "/kaggle/commonlit-models"
    data_dir: str = "/kaggle/input/commonlit-evaluate-student-summaries/"
    use_aug_data: bool = False
    aug_data_dir: str = "/kaggle/input/commonlit-aug-data/"
    gensim_bin_model_path: str = "/kaggle/input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin"
    aug_data_list: [
        "back_translation_Hel_fr"
    ]
    save_to_sheet: str = True
    sheet_json_key: str = '/kaggle/input/ktokunagautils/ktokunaga-4094cf694f5c.json'
    sheet_key: str = '1LhmdqSXborxoP1Pwb1ly-UO_DTfGSfXDN25ZS5MkvHI'
    kaggle_dataset_title: str = "commonlit-models"
    input_cols: List[str] = ["prompt_title", "prompt_question", "fixed_summary_text"]
    lgbm_params = {
        'boosting_type': 'gbdt',
        'random_state': 42,
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.048,
        'max_depth': 4,
        'lambda_l1': 0.0,
        'lambda_l2': 0.011
     }
    additional_features = [
        "summary_length", 
        "splling_err_num", 
        "prompt_length",
        "length_ratio",
        "word_overlap_count",
        "bigram_overlap_count",
        "bigram_overlap_ratio",
        "trigram_overlap_count",
        "trigram_overlap_ratio",
        "quotes_count"
    ]
    report_to: str = "wandb" # none
    on_kaggle: bool = True

class Logger:

    def __init__(self, log_path=''):
        self.general_logger = logging.getLogger('general')
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler(f'{log_path}general.log')
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)

    def info(self, message):
        # 時刻をつけてコンソールとログに出力
        self.general_logger.info('[{}] - {}'.format(self.now_string(), message))

    def now_string(self):
        return str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S'))


class WriteSheet:

    def __init__(self, 
        sheet_json_key,
        sheet_key,
    ):

        import gspread
        
        scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
        credentials = ServiceAccountCredentials.from_json_keyfile_name(sheet_json_key, scope)
        gc = gspread.authorize(credentials)
        self.worksheet = gc.open_by_key(sheet_key)
    

    def write(self, data, sheet_name, table_range='A1'):

        sheet = self.worksheet.worksheet(sheet_name)

        # 辞書のみJSONに変換、ほかはそのままにして、書き込む
        data_json = [json.dumps(d, ensure_ascii=False) if type(d) == dict else d for d in data]
        sheet.append_row(data_json, table_range=table_range)


def get_commit_hash(repo_path='/kaggle/working/kaggle-commonlit/'):

    wd = os.getcwd()
    os.chdir(repo_path)
    
    cmd = "git show --format='%H' --no-patch"
    hash_value = subprocess.check_output(cmd.split()).decode('utf-8')[1:-3]

    os.chdir(wd)

    return hash_value


def class_vars_to_dict(cls):
    return {key: value for key, value in cls.__dict__.items() if not key.startswith("__") and not callable(value)}


def print_gpu_utilization(logger):
    nvmlInit()  #noqa
    handle = nvmlDeviceGetHandleByIndex(0) # noqa
    info = nvmlDeviceGetMemoryInfo(handle) # noqa
    logger.info(f"GPU memory occupied: {info.used//1024**2} MB.")


# set random seed
def seed_everything(seed: int):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class Preprocessor:
    def __init__(self, 
                model_name: str,
                ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(f"{RCFG.base_model_dir}")
        self.twd = TreebankWordDetokenizer()
        self.STOP_WORDS = set(stopwords.words('english'))
        
        self.spacy_ner_model = spacy.load('en_core_web_sm',)
        self.spellchecker = SpellChecker() 

        gensim_model = gensim.models.KeyedVectors.load_word2vec_format(RCFG.gensim_bin_model_path, binary=True)
        self.gensim_words = gensim_model.index_to_key

    def get_probability(self, word): 
        "Probability of `word`."
        # use inverse of rank as proxy
        # returns 0 if the word isn't in the dictionary
        return - self.all_words_rank.get(word, 0)

    def correction(self, word): 
        "Most probable spelling correction for word."
        return max(self.get_candidates(word), key=self.get_probability)

    def get_candidates(self, word): 
        "Generate possible spelling corrections for word."
        return (self.is_known([word]) or self.is_known(self.edits1(word)) or self.is_known(self.edits2(word)) or [word])

    def is_known(self, words): 
        "The subset of `words` that appear in the dictionary of all_words_rank."
        return set(w for w in words if w in self.all_words_rank)

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word): 
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))
    
    def fix_text(self, x):
        return self.twd.detokenize([self.correction(w) for w in x])
        
    def word_overlap_count(self, row):
        """ intersection(prompt_text, text) """        
        def check_is_stop_word(word):
            return word in self.STOP_WORDS
        
        prompt_words = row['prompt_tokens']
        summary_words = row['summary_tokens']
        if self.STOP_WORDS:
            prompt_words = list(filter(check_is_stop_word, prompt_words))
            summary_words = list(filter(check_is_stop_word, summary_words))
        return len(set(prompt_words).intersection(set(summary_words)))
            
    def ngrams(self, token, n):
        # Use the zip function to help us generate n-grams
        # Concatentate the tokens into ngrams and return
        ngrams = zip(*[token[i:] for i in range(n)])
        return [" ".join(ngram) for ngram in ngrams]

    def ngram_co_occurrence(self, row, n: int) -> int:
        # Tokenize the original text and summary into words
        original_tokens = row['prompt_tokens']
        summary_tokens = row['summary_tokens']

        # Generate n-grams for the original text and summary
        original_ngrams = set(self.ngrams(original_tokens, n))
        summary_ngrams = set(self.ngrams(summary_tokens, n))

        # Calculate the number of common n-grams
        common_ngrams = original_ngrams.intersection(summary_ngrams)

        return len(common_ngrams)
    
    def ner_overlap_count(self, row, mode:str):
        model = self.spacy_ner_model
        def clean_ners(ner_list):
            return set([(ner[0].lower(), ner[1]) for ner in ner_list])
        prompt = model(row['prompt_text'])
        summary = model(row['text'])

        if "spacy" in str(model):
            prompt_ner = set([(token.text, token.label_) for token in prompt.ents])
            summary_ner = set([(token.text, token.label_) for token in summary.ents])
        elif "stanza" in str(model):
            prompt_ner = set([(token.text, token.type) for token in prompt.ents])
            summary_ner = set([(token.text, token.type) for token in summary.ents])
        else:
            raise Exception("Model not supported")

        prompt_ner = clean_ners(prompt_ner)
        summary_ner = clean_ners(summary_ner)

        intersecting_ners = prompt_ner.intersection(summary_ner)
        
        ner_dict = dict(Counter([ner[1] for ner in intersecting_ners]))
        
        if mode == "train":
            return ner_dict
        elif mode == "test":
            return {key: ner_dict.get(key) for key in self.ner_keys}

    
    def quotes_count(self, row):
        summary = row['text']
        text = row['prompt_text']
        quotes_from_summary = re.findall(r'"([^"]*)"', summary)
        if len(quotes_from_summary)>0:
            return [quote in text for quote in quotes_from_summary].count(True)
        else:
            return 0

    def spelling(self, text):
        
        wordlist=text.split()
        amount_miss = len(list(self.spellchecker.unknown(wordlist)))

        return amount_miss
    
    def add_spelling_dictionary(self, tokens: List[str]) -> List[str]:
        """dictionary update for pyspell checker and autocorrect"""
        self.spellchecker.word_frequency.load_words(tokens)
    
    def add_all_words(self, series):

        prompt_words = Counter()
        for tokens in series.drop_duplicates():
            prompt_words.update(tokens)
        self.prompt_words_in_order = [item[0] for item in prompt_words.most_common()] 

        self.all_words_rank = {}
        for i,word in enumerate(self.prompt_words_in_order + self.gensim_words):
            if word in self.all_words_rank:
                continue
            self.all_words_rank[word] = i
    
    def run(self, 
            prompts: pd.DataFrame,
            summaries:pd.DataFrame,
            mode:str
        ) -> pd.DataFrame:
        
        # before merge preprocess
        prompts["prompt_length"] = prompts["prompt_text"].apply(
            lambda x: len(word_tokenize(x))
        )
        prompts["prompt_tokens"] = prompts["prompt_text"].apply(
            lambda x: word_tokenize(x)
        )

        summaries["summary_length"] = summaries["text"].apply(
            lambda x: len(word_tokenize(x))
        )
        summaries["summary_tokens"] = summaries["text"].apply(
            lambda x: word_tokenize(x)
        )
        
        # Add prompt tokens into spelling checker dictionary
        prompts["prompt_tokens"].apply(
            lambda x: self.add_spelling_dictionary(x)
        )
        self.add_all_words(prompts['prompt_tokens'])
        
        # fix misspelling
        summaries["fixed_summary_text"] = summaries["summary_tokens"].progress_apply(
            lambda x: self.fix_text(x)
        )
        
        # count misspelling
        summaries["splling_err_num"] = summaries["text"].progress_apply(self.spelling)
        
        # merge prompts and summaries
        input_df = summaries.merge(prompts, how="left", on="prompt_id")

        # after merge preprocess
        input_df['length_ratio'] = input_df['summary_length'] / input_df['prompt_length']
        
        input_df['word_overlap_count'] = input_df.progress_apply(self.word_overlap_count, axis=1)
        input_df['bigram_overlap_count'] = input_df.progress_apply(
            self.ngram_co_occurrence,args=(2,), axis=1 
        )
        input_df['bigram_overlap_ratio'] = input_df['bigram_overlap_count'] / (input_df['summary_length'] - 1)
        
        input_df['trigram_overlap_count'] = input_df.progress_apply(
            self.ngram_co_occurrence, args=(3,), axis=1
        )
        input_df['trigram_overlap_ratio'] = input_df['trigram_overlap_count'] / (input_df['summary_length'] - 2)
        
        input_df['quotes_count'] = input_df.progress_apply(self.quotes_count, axis=1)

        df_features = input_df[RCFG.additional_features].copy()
        
        if mode == 'train':
            scaler = StandardScaler()
            input_df[RCFG.additional_features] = scaler.fit_transform(df_features)
            dump(scaler, open(f"{RCFG.model_dir}/scaler.pkl", "wb"))
        else:
            scaler = load(open(f"{RCFG.model_dir}/scaler.pkl", "rb"))
            input_df[RCFG.additional_features] = scaler.fit_transform(df_features)


        return input_df.drop(columns=["summary_tokens", "prompt_tokens"])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}

def compute_mcrmse(eval_pred):
    """
    Calculates mean columnwise root mean squared error
    https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/overview/evaluation
    """
    preds, labels = eval_pred

    col_rmse = np.sqrt(np.mean((preds - labels) ** 2, axis=0))
    mcrmse = np.mean(col_rmse)

    return {
        "content_rmse": col_rmse[0],
        "wording_rmse": col_rmse[1],
        "mcrmse": mcrmse,
    }

def compt_score(content_true, content_pred, wording_true, wording_pred):
    content_score = mean_squared_error(content_true, content_pred)**(1/2)
    wording_score = mean_squared_error(wording_true, wording_pred)**(1/2)
    
    return (content_score + wording_score)/2

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

class MCRMSELoss(nn.Module):
    def __init__(self, num_scored=2):
        super().__init__()
        self.rmse = RMSELoss()
        self.num_scored = num_scored

    def forward(self, yhat, y):
        score = 0
        for i in range(self.num_scored):
            score += self.rmse(yhat[:, i], y[:, i]) / self.num_scored

        return score

class CustomTransformersModel(nn.Module):
    def __init__(
            self, 
            base_model, 
            num_labels, 
            additional_features_dim, 
            n_freeze = 0, 
            hidden_units=200, 
            dropout=0.2,
            mean_pooling=False
        ):
        super(CustomTransformersModel, self).__init__()
        self.base_model = base_model
        self.additional_features_dim = additional_features_dim

        self.mean_pooling=mean_pooling
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                base_model.config.hidden_size + additional_features_dim, 
                hidden_units,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, num_labels)
        )

        # freezing embeddings layer
        if n_freeze:
            self.base_model.embeddings.requires_grad_(False)
        
            #freezing the initial N layers
            for i in range(0, n_freeze, 1):
                for n,p in self.base_model.encoder.layer[i].named_parameters():
                    p.requires_grad = False

        self.creterion = MCRMSELoss()

    def forward(self, input_ids, features, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)

        if self.mean_pooling:
            # mean pooling
            last_hidden_state = outputs[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            logits = self.classifier(torch.cat((mean_embeddings, features), 1))
        else:
            logits = self.classifier(torch.cat((outputs[0][:, 0, :], features), 1))

        if labels is not None:
            loss = self.creterion(logits, labels)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}
    


class ScoreRegressor:
    def __init__(
        self, 
        model_name: str,
        model_dir: str,
        inputs: List[str],
        target_cols: List[str],
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        max_length: int,
    ):


        self.input_col = "input"        
        self.input_text_cols = inputs 
        self.target_cols = target_cols
        self.additional_feature_cols = RCFG.additional_features

        self.model_name = model_name
        self.model_dir = model_dir
        self.max_length = max_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(f"{RCFG.base_model_dir}")
        self.model_config = AutoConfig.from_pretrained(f"{RCFG.base_model_dir}")
        
        self.model_config.update({
            "hidden_dropout_prob": hidden_dropout_prob,
            "attention_probs_dropout_prob": attention_probs_dropout_prob,
            "num_labels": 2,
            "problem_type": "regression",
        })
        
        seed_everything(seed=42)

        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer
        )

    def concatenate_with_sep_token(self, row):
        sep = " " + self.tokenizer.sep_token + " "        
        return sep.join(row[self.input_text_cols])

    def tokenize_function(self, examples: pd.DataFrame):
        labels = [examples["content"], examples["wording"]]
        features = examples['features']
        tokenized = self.tokenizer(examples[self.input_col],
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length)
        return {
            **tokenized,
            "labels": labels,
            "features": features
        }
    
    def tokenize_function_test(self, examples: pd.DataFrame):
        tokenized = self.tokenizer(examples[self.input_col],
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length)
        return tokenized
        
    def train(self, 
            fold: int,
            train_df: pd.DataFrame,
            valid_df: pd.DataFrame,
            batch_size: int,
            learning_rate: float,
            weight_decay: float,
            num_train_epochs: float,
            save_steps: int,
        ) -> None:
        """fine-tuning"""
        
        train_df[self.input_col] = train_df.apply(self.concatenate_with_sep_token, axis=1)
        valid_df[self.input_col] = valid_df.apply(self.concatenate_with_sep_token, axis=1) 
        
        train_df['features'] = train_df[self.additional_feature_cols].to_numpy().tolist()
        valid_df['features'] = valid_df[self.additional_feature_cols].to_numpy().tolist()
        train_df = train_df[[self.input_col] + ['features'] + self.target_cols]
        valid_df = valid_df[[self.input_col] + ['features'] +  self.target_cols]

        
        model_content = AutoModel.from_pretrained(
            f"{RCFG.base_model_dir}", 
            config=self.model_config
        )

        custom_model = CustomTransformersModel(
            model_content, 
            num_labels=2, 
            additional_features_dim=len(self.additional_feature_cols),
            n_freeze=CFG.n_freeze,
            mean_pooling=CFG.mean_pooling
        )

        train_dataset = Dataset.from_pandas(train_df, preserve_index=False) 
        val_dataset = Dataset.from_pandas(valid_df, preserve_index=False) 
    
        train_tokenized_datasets = train_dataset.map(self.tokenize_function, batched=False)
        val_tokenized_datasets = val_dataset.map(self.tokenize_function, batched=False)

        # eg. "bert/fold_0/"
        model_fold_dir = os.path.join(self.model_dir, str(fold)) 
        
        training_args = TrainingArguments(
            output_dir=model_fold_dir,
            load_best_model_at_end=True, # select best model
            learning_rate=learning_rate,
            per_gpu_train_batch_size=batch_size,
            # gradient_accumulation_steps=4,
            # per_device_train_batch_size=3, # batch_sizeは12 = 3 * 4
            per_device_eval_batch_size=8,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            report_to=RCFG.report_to,
            greater_is_better=False,
            save_strategy="steps",
            evaluation_strategy="steps",
            eval_steps=save_steps,
            save_steps=save_steps,
            metric_for_best_model="mcrmse",
            fp16=True,
            save_total_limit=1
            # gradient_checkpointing=True
        )

        trainer = Trainer(
            model=custom_model,
            args=training_args,
            train_dataset=train_tokenized_datasets,
            eval_dataset=val_tokenized_datasets,
            tokenizer=self.tokenizer,
            compute_metrics=compute_mcrmse,
            data_collator=self.data_collator
        )

        trainer.train()
        
        torch.save(custom_model, os.path.join(self.model_dir, "model_weight.pth"))
        self.tokenizer.save_pretrained(self.model_dir)

        custom_model.cpu()
        del custom_model
        gc.collect()
        torch.cuda.empty_cache()
    
        
    def predict(self, 
                test_df: pd.DataFrame,
                batch_size: int,
                fold: int,
               ):
        """predict content score"""
        
        test_df[self.input_col] = test_df.apply(self.concatenate_with_sep_token, axis=1)

        test_df['features'] = test_df[self.additional_feature_cols].to_numpy().tolist()
        test_df = test_df[[self.input_col] + ['features']]

        test_ = test_df[[self.input_col] + ['features']]

        test_dataset = Dataset.from_pandas(test_, preserve_index=False) 
        test_tokenized_dataset = test_dataset.map(self.tokenize_function_test, batched=False)

        custom_model = torch.load(os.path.join(self.model_dir, "model_weight.pth"))
        custom_model.eval()
        
        model_fold_dir = os.path.join(self.model_dir, str(fold)) 

        test_args = TrainingArguments(
            output_dir=model_fold_dir,
            do_train = False,
            do_predict = True,
            per_device_eval_batch_size=batch_size,
            dataloader_drop_last = False,
        )

        # init trainer
        infer_content = Trainer(
                      model = custom_model, 
                      tokenizer=self.tokenizer,
                      data_collator=self.data_collator,
                      args = test_args)

        preds = infer_content.predict(test_tokenized_dataset)[0]
        pred_df = pd.DataFrame(
                    preds, 
                    columns=[
                        f"content_pred", 
                        f"wording_pred"
                    ]
                )

        custom_model.cpu()
        del custom_model
        gc.collect()
        torch.cuda.empty_cache()

        return pred_df


def train_by_fold(
        logger,
        train_df: pd.DataFrame,
        model_name: str,
        targets: List[str],
        inputs: List[str],
        batch_size: int,
        learning_rate: int,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        weight_decay: float,
        num_train_epochs: int,
        save_steps: int,
        max_length:int,
        df_augtrain = None
    ):

    model_dir =  f"{RCFG.model_dir}/{model_name}"
    logger.info(f'training model dir: {model_dir}.')

    # delete old model files
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
        
    for fold in range(CFG.n_splits):
        logger.info(f"fold {fold}:")
        
        train_data = train_df[train_df["fold"] != fold]
        valid_data = train_df[train_df["fold"] == fold]

        if RCFG.use_aug_data: 
            logger.info('Augment data by back translation.')
            train_aug_data = df_augtrain[df_augtrain["fold"] != fold]
            train_data = pd.concat([train_data, train_aug_data])
        
        fold_model_dir = f'{model_dir}/fold_{fold}'
        csr = ScoreRegressor(
            model_name=model_name,
            target_cols=targets,
            inputs= inputs,
            model_dir = fold_model_dir, 
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_length=max_length,
        )
        
        csr.train(
            fold=fold,
            train_df=train_data,
            valid_df=valid_data, 
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_train_epochs=num_train_epochs,
            save_steps=save_steps,
        )

        del csr
        torch.cuda.empty_cache()
        print_gpu_utilization(logger)

def validate(
    logger,
    train_df: pd.DataFrame,
    targets: List[str],
    inputs: List[str],
    batch_size: int,
    model_name: str,
    hidden_dropout_prob: float,
    attention_probs_dropout_prob: float,
    max_length : int
) -> pd.DataFrame:
    """predict oof data"""

    columns = list(train_df.columns.values)
    
    for fold in range(CFG.n_splits):
        print(f"fold {fold}:")
        
        valid_data = train_df[train_df["fold"] == fold]
        
        model_dir =  f"{RCFG.model_dir}/{model_name}/fold_{fold}"
        
        csr = ScoreRegressor(
            model_name=model_name,
            target_cols=targets,
            inputs= inputs,
            model_dir = model_dir,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_length=max_length,
           )
        
        pred_df = csr.predict(
            test_df=valid_data, 
            batch_size=batch_size,
            fold=fold
        )

        train_df.loc[valid_data.index, f"content_multi_pred"] = pred_df[f"content_pred"].values
        train_df.loc[valid_data.index, f"wording_multi_pred"] = pred_df[f"wording_pred"].values
                
    return train_df[columns + [f"content_multi_pred", f"wording_multi_pred"]]

def predict(
    logger,
    test_df: pd.DataFrame,
    targets:List[str],
    inputs: List[str],
    batch_size: int,
    model_name: str,
    hidden_dropout_prob: float,
    attention_probs_dropout_prob: float,
    max_length : int
    ):
    """predict using mean folds"""
    
    columns = list(test_df.columns.values)

    for fold in range(CFG.n_splits):
        logger.info(f"fold {fold}:")
        
        model_dir =  f"{RCFG.model_dir}/{model_name}/fold_{fold}"
        logger.info(f'prediction model dir: {model_dir}.')

        csr = ScoreRegressor(
            model_name=model_name,
            target_cols=targets,
            inputs= inputs,
            model_dir = model_dir, 
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_length=max_length,
           )
        
        pred_df = csr.predict(
            test_df=test_df, 
            batch_size=batch_size,
            fold=fold
        )

        test_df[f"content_multi_pred_{fold}"] = pred_df[f"content_pred"].values
        test_df[f"wording_multi_pred_{fold}"] = pred_df[f"wording_pred"].values

    test_df[f"content_multi_pred"] = test_df[[f"content_multi_pred_{fold}" for fold in range(CFG.n_splits)]].mean(axis=1)
    test_df[f"wording_multi_pred"] = test_df[[f"wording_multi_pred_{fold}" for fold in range(CFG.n_splits)]].mean(axis=1)
    
    return test_df[columns + [f"content_multi_pred", f"wording_multi_pred"]]


class Runner():

    def __init__(
        self,
    ):

        warnings.simplefilter("ignore")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
        disable_progress_bar()
        tqdm.pandas()

        seed_everything(seed=42)
        transformers.logging.set_verbosity_error()

        self.targets = ["content", "wording"]
        self.logger = Logger(RCFG.output_path)

        self.data_to_write = []

        if RCFG.save_to_sheet:
            self.logger.info('Initializing Google Sheet.')
            self.sheet = WriteSheet(
                sheet_json_key = RCFG.sheet_json_key,
                sheet_key = RCFG.sheet_key
            )

        if RCFG.report_to == 'wandb':

            if RCFG.on_kaggle:
                from kaggle_secrets import UserSecretsClient
                user_secrets = UserSecretsClient()
                secret_value_0 = user_secrets.get_secret("wandb_api_key")
            else:
                secret_value_0 = RCFG.wandb_api_key
            
            import wandb
            wandb.login(key=secret_value_0)
            run = wandb.init(
                project='commonlit', 
                name=RCFG.run_name,
                config=class_vars_to_dict(CFG),
                group=CFG.model_name,
                job_type="train"
            )

    def load_dataset(self):

        self.prompts_train = pd.read_csv(RCFG.data_dir + "prompts_train.csv")
        self.prompts_test = pd.read_csv(RCFG.data_dir + "prompts_test.csv")
        self.summaries_train = pd.read_csv(RCFG.data_dir + "summaries_train.csv")
        self.summaries_test = pd.read_csv(RCFG.data_dir + "summaries_test.csv")
        self.sample_submission = pd.read_csv(RCFG.data_dir + "sample_submission.csv")

        if RCFG.debug:
            self.logger.info('Debug mode. Reduce train data.')
            self.summaries_train = self.summaries_train.head(RCFG.debug_size) # for dev mode
        
        self.augtrain = None
        if RCFG.use_aug_data:
            
            self.augtrain = pd.read_csv(RCFG.aug_data_dir + "back_translation_all.csv")
            self.augtrain = self.augtrain[self.augtrain['lang'].isin(RCFG.aug_data_list)].drop(['lang'], axis=1)
            self.augtrain.columns = ['student_id', 'fixed_summary_text']

    def preprocess(self):

        self.logger.info('Start Preprocess.')
        preprocessor = Preprocessor(model_name=CFG.model_name)

        if RCFG.train:
            self.logger.info('Preprocess train data.')
            self.train = preprocessor.run(self.prompts_train, self.summaries_train, mode="train")
        
        if RCFG.predict:
            self.logger.info('Preprocess test data.')
            self.test = preprocessor.run(self.prompts_test, self.summaries_test, mode="test")


    def run_transformers_regressor(self):

        gkf = GroupKFold(n_splits=CFG.n_splits)
        if RCFG.train:
            for i, (_, val_index) in enumerate(gkf.split(self.train, groups=self.train["prompt_id"])):
                self.train.loc[val_index, "fold"] = i
        
        if RCFG.use_aug_data:
            self.logger.info('Use augmented data.')

            df_master = self.train.copy().drop(['fixed_summary_text'], axis=1)
            self.augtrain = self.augtrain.merge(df_master, on="student_id", how="left")
            self.augtrain = self.augtrain[self.augtrain['prompt_id'].notnull()]
            self.augtrain = self.augtrain[self.train.columns]

        input_cols = RCFG.input_cols

        if RCFG.train:
            
            torch.cuda.empty_cache()
            print_gpu_utilization(self.logger) # 2, 7117　(2, 6137)
            self.logger.info(f'Start training by fold.')
            
            train_by_fold(
                logger=self.logger,
                train_df=self.train,
                model_name=CFG.model_name,
                targets=self.targets,
                inputs=input_cols,
                learning_rate=CFG.learning_rate,
                hidden_dropout_prob=CFG.hidden_dropout_prob,
                attention_probs_dropout_prob=CFG.attention_probs_dropout_prob,
                weight_decay=CFG.weight_decay,
                num_train_epochs=CFG.num_train_epochs,
                batch_size=CFG.batch_size,
                save_steps=CFG.save_steps,
                max_length=CFG.max_length,
                df_augtrain = self.augtrain
            )
            
            print_gpu_utilization(self.logger) # 7117, 6739 (1719, 1719)
            self.logger.info(f'Start creating oof prediction.')
            self.train = validate(
                logger=self.logger,
                train_df=self.train,
                targets=self.targets,
                inputs=input_cols,
                batch_size=CFG.batch_size,
                model_name=CFG.model_name,
                hidden_dropout_prob=CFG.hidden_dropout_prob,
                attention_probs_dropout_prob=CFG.attention_probs_dropout_prob,
                max_length=CFG.max_length
            )

            # set validate result
            rmses = []
            for target in self.targets:
                rmse = mean_squared_error(self.train[target], self.train[f"{target}_multi_pred"], squared=False)
                print(f"cv {target} rmse: {rmse}")
                self.logger.info(f"cv {target} rmse: {rmse}")
                self.data_to_write.append(rmse)
                rmses.append(rmse)
            self.data_to_write.append(np.mean(rmses))
        
        if RCFG.predict:
            
            print_gpu_utilization(self.logger) # 7117, 6739 (3907, 3907)
            self.logger.info(f'Start Predicting.')
            self.test = predict(
                logger=self.logger,
                test_df=self.test,
                targets=self.targets,
                inputs = input_cols,
                batch_size=CFG.batch_size,
                model_name=CFG.model_name,
                hidden_dropout_prob=CFG.hidden_dropout_prob,
                attention_probs_dropout_prob=CFG.attention_probs_dropout_prob,
                max_length=CFG.max_length
            )


            print_gpu_utilization(self.logger) # 7117, 7115 (6137, 6137)
        
        if RCFG.train:
            self.train.to_csv(f'{RCFG.model_dir}/train_processed.csv', index=False)
        
        if RCFG.report_to == 'wandb':
            
            import wandb
            wandb.finish()

    def create_prediction(self):

        if not RCFG.predict:
            return None

        self.logger.info('Start creating submission data.')
        df_output = self.test[["student_id", "content_multi_pred", "wording_multi_pred"]]
        df_output = df_output.rename(columns={"content_multi_pred": "content", "wording_multi_pred": "wording"})

        df_output.to_csv("submission.csv", index=False)


    def write_sheet(self, ):
        self.logger.info('Write scores to google sheet.')

        nowstr_jst = str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S'))
        base_data = [nowstr_jst, RCFG.commit_hash, class_vars_to_dict(CFG),  class_vars_to_dict(RCFG)]
        self.data_to_write = base_data + self.data_to_write
        self.sheet.write(self.data_to_write, sheet_name='cvscores')


    def save_model_as_kaggle_dataset(self,):

        self.logger.info(f'Save {RCFG.model_dir} as kaggle dataset.')
        metadata = {
            "title": RCFG.kaggle_dataset_title,
            "id": f"kazuakitokunaga/{RCFG.kaggle_dataset_title}",
            "licenses": [
                {
                "name": "CC0-1.0"
                }
            ]
            }

        with open(f'{RCFG.model_dir}/dataset-metadata.json', 'w') as f:
            json.dump(metadata, f)

        subprocess.call(f'kaggle datasets version -r zip -p {RCFG.model_dir} -m "Updateddata"'.split())