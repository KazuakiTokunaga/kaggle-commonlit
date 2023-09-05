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
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from datasets import Dataset,load_dataset, load_from_disk
from transformers import TrainingArguments, Trainer
from datasets import load_metric, disable_progress_bar
from sklearn.metrics import mean_squared_error
import torch
from sklearn.model_selection import KFold, GroupKFold
from oauth2client.service_account import ServiceAccountCredentials
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from collections import Counter
import spacy
import re
from autocorrect import Speller
from spellchecker import SpellChecker
import lightgbm as lgb

class CFG:
    model_name: str ="debertav3base"
    learning_rate: float =0.000016
    weight_decay: float =0.03
    hidden_dropout_prob: float =0.007
    attention_probs_dropout_prob: float =0.007
    num_train_epochs: int =5
    n_splits: int =4
    batch_size: int =12
    random_seed: int =42
    save_steps: int =100
    max_length: int =512

class RCFG:
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
    aug_data_list: [
        "back_translation_Hel_fr"
    ]
    save_to_sheet: str = True
    sheet_json_key: str = '/kaggle/input/ktokunagautils/ktokunaga-4094cf694f5c.json'
    sheet_key: str = '1LhmdqSXborxoP1Pwb1ly-UO_DTfGSfXDN25ZS5MkvHI'
    kaggle_dataset_title: str = "commonlit-models"
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
        self.speller = Speller(lang='en')
        self.spellchecker = SpellChecker() 
        
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

        # # Optionally, you can get the frequency of common n-grams for a more nuanced analysis
        # original_ngram_freq = Counter(ngrams(original_words, n))
        # summary_ngram_freq = Counter(ngrams(summary_words, n))
        # common_ngram_freq = {ngram: min(original_ngram_freq[ngram], summary_ngram_freq[ngram]) for ngram in common_ngrams}

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
        self.speller.nlp_data.update({token:1000 for token in tokens})
    
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
        
#         from IPython.core.debugger import Pdb; Pdb().set_trace()
        # fix misspelling
        summaries["fixed_summary_text"] = summaries["text"].progress_apply(
            lambda x: self.speller(x)
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
        
        # Crate dataframe with count of each category NERs overlap for all the summaries
        # Because it spends too much time for this feature, I don't use this time.
#         ners_count_df  = input_df.progress_apply(
#             lambda row: pd.Series(self.ner_overlap_count(row, mode=mode), dtype='float64'), axis=1
#         ).fillna(0)
#         self.ner_keys = ners_count_df.columns
#         ners_count_df['sum'] = ners_count_df.sum(axis=1)
#         ners_count_df.columns = ['NER_' + col for col in ners_count_df.columns]
#         # join ner count dataframe with train dataframe
#         input_df = pd.concat([input_df, ners_count_df], axis=1)
        
        input_df['quotes_count'] = input_df.progress_apply(self.quotes_count, axis=1)
        
        return input_df.drop(columns=["summary_tokens", "prompt_tokens"])


def compute_rmse_loss(outputs, targets):
    mse_loss = torch.mean((outputs - targets) ** 2, dim=0)  # Compute MSE for each dimension
    rmse_loss = torch.sqrt(mse_loss)  # Compute RMSE for each dimension
    return torch.mean(rmse_loss)

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
        tokenized = self.tokenizer(examples[self.input_col],
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length)
        return {
            **tokenized,
            "labels": labels,
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
        
        sep = self.tokenizer.sep_token
        train_df[self.input_col] = train_df.apply(self.concatenate_with_sep_token, axis=1)
        valid_df[self.input_col] = valid_df.apply(self.concatenate_with_sep_token, axis=1) 
        
        train_df = train_df[[self.input_col] + self.target_cols]
        valid_df = valid_df[[self.input_col] + self.target_cols]
        
        model_content = AutoModelForSequenceClassification.from_pretrained(
            f"{RCFG.base_model_dir}", 
            config=self.model_config
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
            # per_gpu_train_batch_size=batch_size
            # gradient_accumulation_steps=4,
            per_device_train_batch_size=12, # batch_sizeは12 = 3 * 4
            per_device_eval_batch_size=8,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            report_to='none',
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
            model=model_content,
            args=training_args,
            train_dataset=train_tokenized_datasets,
            eval_dataset=val_tokenized_datasets,
            tokenizer=self.tokenizer,
            compute_metrics=compute_mcrmse,
            data_collator=self.data_collator
        )
        trainer.compute_loss = compute_rmse_loss

        trainer.train()
        
        model_content.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)

        model_content.cpu()
        del model_content
        gc.collect()
        torch.cuda.empty_cache()
    
        
    def predict(self, 
                test_df: pd.DataFrame,
                batch_size: int,
                fold: int,
               ):
        """predict content score"""
        
        sep = self.tokenizer.sep_token
        test_df[self.input_col] = test_df.apply(self.concatenate_with_sep_token, axis=1)

        test_ = test_df[[self.input_col]]
    
        test_dataset = Dataset.from_pandas(test_, preserve_index=False) 
        test_tokenized_dataset = test_dataset.map(self.tokenize_function_test, batched=False)

        model_content = AutoModelForSequenceClassification.from_pretrained(f"{self.model_dir}")
        model_content.eval()
        
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
                      model = model_content, 
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

        model_content.cpu()
        del model_content
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
            df_master = self.train[['student_id', 'prompt_id', 'prompt_title', 'prompt_question', 'content', 'wording', 'fold']]
            self.augtrain = self.augtrain.merge(df_master, on="student_id", how="left")
            self.augtrain = self.augtrain[self.augtrain['prompt_id'].notnull()]

        input_cols = ["prompt_title", "prompt_question", "fixed_summary_text"]

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
            for target in self.targets:
                rmse = mean_squared_error(self.train[target], self.train[f"{target}_multi_pred"], squared=False)
                print(f"cv {target} rmse: {rmse}")
            self.logger.info(f"cv {target} rmse: {rmse}")
            self.data_to_write.append(rmse)
        
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

    def run_lgbm(self):

        if not RCFG.train:
            return None
        
        drop_columns = ["fold", "student_id", "prompt_id", "text", "fixed_summary_text",
                        "prompt_question", "prompt_title", 
                        "prompt_text"
                    ] + self.targets
        
        self.model_dict = {}

        for target in self.targets:
            self.logger.info(f'Start training LGBM model: {target}')

            models = []
            for fold in range(CFG.n_splits):

                X_train_cv = self.train[self.train["fold"] != fold].drop(columns=drop_columns)
                y_train_cv = self.train[self.train["fold"] != fold][target]

                X_eval_cv = self.train[self.train["fold"] == fold].drop(columns=drop_columns)
                y_eval_cv = self.train[self.train["fold"] == fold][target]

                dtrain = lgb.Dataset(X_train_cv, label=y_train_cv)
                dval = lgb.Dataset(X_eval_cv, label=y_eval_cv)

                evaluation_results = {}
                model = lgb.train(
                    RCFG.lgbm_params,
                    num_boost_round=10000,
                        #categorical_feature = categorical_features,
                    valid_names=['train', 'valid'],
                    train_set=dtrain,
                    valid_sets=dval,
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=30, verbose=False),
                        lgb.log_evaluation(-1),
                        lgb.callback.record_evaluation(evaluation_results)
                    ],
                )
                models.append(model)
            
            self.model_dict[target] = models

        # cv
        rmses = []

        for target in self.targets:
            models = self.model_dict[target]

            preds = []
            trues = []
            
            for fold, model in enumerate(models):
                # ilocで取り出す行を指定
                X_eval_cv = self.train[self.train["fold"] == fold].drop(columns=drop_columns)
                y_eval_cv = self.train[self.train["fold"] == fold][target]

                pred = model.predict(X_eval_cv)

                trues.extend(y_eval_cv)
                preds.extend(pred)
                
            rmse = np.sqrt(mean_squared_error(trues, preds))
            self.logger.info(f"{target}_rmse : {rmse}")
            self.data_to_write.append(rmse)
            rmses = rmses + [rmse]

        mcrmse = sum(rmses) / len(rmses)
        self.logger.info(f"mcrmse : {mcrmse}")
        self.data_to_write.append(mcrmse)


        # delete old model files
        model_dir = f'{RCFG.model_dir}/gbtmodel'
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)

        save_model_path = f'{model_dir}/model_dict.pkl'
        self.logger.info(f'save LGBM model: {save_model_path}')
        with open(save_model_path, 'wb') as f:
            pickle.dump(self.model_dict, f)


    def create_prediction(self):

        if not RCFG.predict:
            return None

        self.logger.info('Start creating submission data using LGBM.')
        with open(f'{RCFG.model_dir}/gbtmodel/model_dict.pkl', 'rb') as f:
            self.model_dict = pickle.load(f)

        drop_columns = [
                        #"fold", 
                        "student_id", "prompt_id", "text",  "fixed_summary_text",
                        "prompt_question", "prompt_title", 
                        "prompt_text",
                    ]

        pred_dict = {}
        for target in self.targets:
            models = self.model_dict[target]
            preds = []

            for fold, model in enumerate(models):
                # ilocで取り出す行を指定
                X_eval_cv = self.test.drop(columns=drop_columns)

                pred = model.predict(X_eval_cv)
                preds.append(pred)
            
            pred_dict[target] = preds

        for target in self.targets:
            preds = pred_dict[target]
            for i, pred in enumerate(preds):
                self.test[f"{target}_pred_{i}"] = pred

            self.test[target] = self.test[[f"{target}_pred_{fold}" for fold in range(CFG.n_splits)]].mean(axis=1)

        self.test[["student_id", "content", "wording"]].to_csv("submission.csv", index=False)


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