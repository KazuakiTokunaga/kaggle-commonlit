from typing import List
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
import logging
import os
import random
import shutil
import json
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from datasets import Dataset,load_dataset, load_from_disk
from transformers import TrainingArguments, Trainer
from datasets import load_metric, disable_progress_bar
from sklearn.metrics import mean_squared_error
import torch
from sklearn.model_selection import KFold, GroupKFold
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from collections import Counter
import spacy
import re
from spellchecker import SpellChecker
import lightgbm as lgb


# set random seed
def seed_everything(seed: int):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class CFG:
    model_name="debertav3base"
    learning_rate=1.5e-5
    weight_decay=0.02
    hidden_dropout_prob=0.005
    attention_probs_dropout_prob=0.005
    num_train_epochs=5
    n_splits=4
    batch_size=12
    random_seed=42
    save_steps=100
    max_length=512
    debug=True
    debug_size=10


class Preprocessor:
    def __init__(self, 
                model_name: str,
                ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(f"/kaggle/input/{model_name}")
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

class ContentScoreRegressor:
    def __init__(self, 
                model_name: str,
                model_dir: str,
                target: str,
                hidden_dropout_prob: float,
                attention_probs_dropout_prob: float,
                max_length: int,
                ):
        self.inputs = ["prompt_text", "prompt_title", "prompt_question", "fixed_summary_text"]
        self.input_col = "input"
        
        self.text_cols = [self.input_col] 
        self.target = target
        self.target_cols = [target]

        self.model_name = model_name
        self.model_dir = model_dir
        self.max_length = max_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(f"/kaggle/input/{model_name}")
        self.model_config = AutoConfig.from_pretrained(f"/kaggle/input/{model_name}")
        
        self.model_config.update({
            "hidden_dropout_prob": hidden_dropout_prob,
            "attention_probs_dropout_prob": attention_probs_dropout_prob,
            "num_labels": 1,
            "problem_type": "regression",
        })
        
        seed_everything(seed=42)

        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer
        )


    def tokenize_function(self, examples: pd.DataFrame):
        labels = [examples[self.target]]
        tokenized = self.tokenizer(examples[self.input_col],
                         padding=False,
                         truncation=True,
                         max_length=self.max_length)
        return {
            **tokenized,
            "labels": labels,
        }
    
    def tokenize_function_test(self, examples: pd.DataFrame):
        tokenized = self.tokenizer(examples[self.input_col],
                         padding=False,
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
        train_df[self.input_col] = (
                    train_df["prompt_title"] + sep 
                    + train_df["prompt_question"] + sep 
                    + train_df["fixed_summary_text"]
                  )

        valid_df[self.input_col] = (
                    valid_df["prompt_title"] + sep 
                    + valid_df["prompt_question"] + sep 
                    + valid_df["fixed_summary_text"]
                  )
        
        train_df = train_df[[self.input_col] + self.target_cols]
        valid_df = valid_df[[self.input_col] + self.target_cols]
        
        model_content = AutoModelForSequenceClassification.from_pretrained(
            f"/kaggle/input/{self.model_name}", 
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
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=8,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            report_to='none',
            greater_is_better=False,
            save_strategy="steps",
            evaluation_strategy="steps",
            eval_steps=save_steps,
            save_steps=save_steps,
            metric_for_best_model="rmse",
            save_total_limit=1
        )

        trainer = Trainer(
            model=model_content,
            args=training_args,
            train_dataset=train_tokenized_datasets,
            eval_dataset=val_tfdropokenized_datasets,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            data_collator=self.data_collator
        )

        trainer.train()
        
        model_content.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)

        
    def predict(self, 
                test_df: pd.DataFrame,
                fold: int,
               ):
        """predict content score"""
        
        sep = self.tokenizer.sep_token
        in_text = (
                    test_df["prompt_title"] + sep 
                    + test_df["prompt_question"] + sep 
                    + test_df["fixed_summary_text"]
                  )
        test_df[self.input_col] = in_text

        test_ = test_df[[self.input_col]]
    
        test_dataset = Dataset.from_pandas(test_, preserve_index=False) 
        test_tokenized_dataset = test_dataset.map(self.tokenize_function_test, batched=False)

        model_content = AutoModelForSequenceClassification.from_pretrained(f"{self.model_dir}")
        model_content.eval()
        
        # e.g. "bert/fold_0/"
        model_fold_dir = os.path.join(self.model_dir, str(fold)) 

        test_args = TrainingArguments(
            output_dir=model_fold_dir,
            do_train = False,
            do_predict = True,
            per_device_eval_batch_size = 4,   
            dataloader_drop_last = False,
        )

        # init trainer
        infer_content = Trainer(
                      model = model_content, 
                      tokenizer=self.tokenizer,
                      data_collator=self.data_collator,
                      args = test_args)

        preds = infer_content.predict(test_tokenized_dataset)[0]

        return preds


def train_by_fold(
        train_df: pd.DataFrame,
        model_name: str,
        target:str,
        save_each_model: bool,
        n_splits: int,
        batch_size: int,
        learning_rate: int,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        weight_decay: float,
        num_train_epochs: int,
        save_steps: int,
        max_length:int
    ):

    # delete old model files
    if os.path.exists(model_name):
        shutil.rmtree(model_name)
    
    os.mkdir(model_name)
        
    for fold in range(CFG.n_splits):
        print(f"fold {fold}:")
        
        train_data = train_df[train_df["fold"] != fold]
        valid_data = train_df[train_df["fold"] == fold]
        
        if save_each_model == True:
            model_dir =  f"{target}/{model_name}/fold_{fold}"
        else: 
            model_dir =  f"{model_name}/fold_{fold}"

        csr = ContentScoreRegressor(
            model_name=model_name,
            target=target,
            model_dir = model_dir, 
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

def validate(
    train_df: pd.DataFrame,
    target:str,
    save_each_model: bool,
    model_name: str,
    hidden_dropout_prob: float,
    attention_probs_dropout_prob: float,
    max_length : int
    ) -> pd.DataFrame:
    """predict oof data"""
    for fold in range(CFG.n_splits):
        print(f"fold {fold}:")
        
        valid_data = train_df[train_df["fold"] == fold]
        
        if save_each_model == True:
            model_dir =  f"{target}/{model_name}/fold_{fold}"
        else: 
            model_dir =  f"{model_name}/fold_{fold}"
        
        csr = ContentScoreRegressor(
            model_name=model_name,
            target=target,
            model_dir = model_dir, # モデル・foldごとにモデルファイルの保存先のdirを分ける
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_length=max_length,
           )
        
        pred = csr.predict(
            test_df=valid_data, 
            fold=fold
        )
        
        train_df.loc[valid_data.index, f"{target}_pred"] = pred

    return train_df
    
def predict(
    test_df: pd.DataFrame,
    target:str,
    save_each_model: bool,
    model_name: str,
    hidden_dropout_prob: float,
    attention_probs_dropout_prob: float,
    max_length : int
    ):
    """predict using mean folds"""

    for fold in range(CFG.n_splits):
        print(f"fold {fold}:")
        
        if save_each_model == True:
            model_dir =  f"{target}/{model_name}/fold_{fold}"
        else: 
            model_dir =  f"{model_name}/fold_{fold}"

        csr = ContentScoreRegressor(
            model_name=model_name,
            target=target,
            model_dir = model_dir, 
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_length=max_length,
           )
        
        pred = csr.predict(
            test_df=test_df, 
            fold=fold
        )
        
        test_df[f"{target}_pred_{fold}"] = pred
    
    test_df[f"{target}"] = test_df[[f"{target}_pred_{fold}" for fold in range(CFG.n_splits)]].mean(axis=1)

    return test_df


class Runner():

    def __init__(
        self,
    ):
        pass

    def run(self):

        warnings.simplefilter("ignore")
        logging.disable(logging.ERROR)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
        disable_progress_bar()
        tqdm.pandas()

        seed_everything(seed=42)

        DATA_DIR = "/kaggle/input/commonlit-evaluate-student-summaries/"

        self.prompts_train = pd.read_csv(DATA_DIR + "prompts_train.csv")
        self.prompts_test = pd.read_csv(DATA_DIR + "prompts_test.csv")
        self.summaries_train = pd.read_csv(DATA_DIR + "summaries_train.csv")
        self.summaries_test = pd.read_csv(DATA_DIR + "summaries_test.csv")
        self.sample_submission = pd.read_csv(DATA_DIR + "sample_submission.csv")

        if CFG.debug:
            self.summaries_train = self.summaries_train.head(CFG.debug_size) # for dev mode

        preprocessor = Preprocessor(model_name=CFG.model_name)

        self.train = preprocessor.run(self.prompts_train, self.summaries_train, mode="train")
        self.test = preprocessor.run(self.prompts_test, self.summaries_test, mode="test")

        gkf = GroupKFold(n_splits=CFG.n_splits)

        for i, (_, val_index) in enumerate(gkf.split(self.train, groups=self.train["prompt_id"])):
            self.train.loc[val_index, "fold"] = i

        for target in ["content", "wording"]:
            train_by_fold(
                self.train,
                model_name=CFG.model_name,
                save_each_model=False,
                target=target,
                learning_rate=CFG.learning_rate,
                hidden_dropout_prob=CFG.hidden_dropout_prob,
                attention_probs_dropout_prob=CFG.attention_probs_dropout_prob,
                weight_decay=CFG.weight_decay,
                num_train_epochs=CFG.num_train_epochs,
                n_splits=CFG.n_splits,
                batch_size=CFG.batch_size,
                save_steps=CFG.save_steps,
                max_length=CFG.max_length
            )
            
            
            self.train = validate(
                self.train,
                target=target,
                save_each_model=False,
                model_name=CFG.model_name,
                hidden_dropout_prob=CFG.hidden_dropout_prob,
                attention_probs_dropout_prob=CFG.attention_probs_dropout_prob,
                max_length=CFG.max_length
            )

            rmse = mean_squared_error(self.train[target], self.train[f"{target}_pred"], squared=False)
            print(f"cv {target} rmse: {rmse}")

            self.test = predict(
                self.test,
                target=target,
                save_each_model=False,
                model_name=CFG.model_name,
                hidden_dropout_prob=CFG.hidden_dropout_prob,
                attention_probs_dropout_prob=CFG.attention_probs_dropout_prob,
                max_length=CFG.max_length
            )

        targets = ["content", "wording"]

        drop_columns = ["fold", "student_id", "prompt_id", "text", "fixed_summary_text",
                        "prompt_question", "prompt_title", 
                        "prompt_text"
                    ] + targets

        model_dict = {}

        for target in targets:
            models = []
            
            for fold in range(CFG.n_splits):

                X_train_cv = self.train[self.train["fold"] != fold].drop(columns=drop_columns)
                y_train_cv = self.train[self.train["fold"] != fold][target]

                X_eval_cv = self.train[self.train["fold"] == fold].drop(columns=drop_columns)
                y_eval_cv = self.train[self.train["fold"] == fold][target]

                dtrain = lgb.Dataset(X_train_cv, label=y_train_cv)
                dval = lgb.Dataset(X_eval_cv, label=y_eval_cv)

                params = {
                        'boosting_type': 'gbdt',
                        'random_state': 42,
                        'objective': 'regression',
                        'metric': 'rmse',
                        'learning_rate': 0.05,
                        }

                evaluation_results = {}
                model = lgb.train(params,
                                num_boost_round=10000,
                                    #categorical_feature = categorical_features,
                                valid_names=['train', 'valid'],
                                train_set=dtrain,
                                valid_sets=dval,
                                callbacks=[
                                    lgb.early_stopping(stopping_rounds=30, verbose=True),
                                    lgb.log_evaluation(100),
                                    lgb.callback.record_evaluation(evaluation_results)
                                    ],
                                )
                models.append(model)
            
            model_dict[target] = models

        # cv
        rmses = []

        for target in targets:
            models = model_dict[target]

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
            print(f"{target}_rmse : {rmse}")
            rmses = rmses + [rmse]

        print(f"mcrmse : {sum(rmses) / len(rmses)}")

        drop_columns = [
                        #"fold", 
                        "student_id", "prompt_id", "text", 
                        "prompt_question", "prompt_title", 
                        "prompt_text",
                        "input"
                    ] + [
                        f"content_pred_{i}" for i in range(CFG.n_splits)
                        ] + [
                        f"wording_pred_{i}" for i in range(CFG.n_splits)
                        ]

        pred_dict = {}
        for target in targets:
            models = model_dict[target]
            preds = []

            for fold, model in enumerate(models):
                # ilocで取り出す行を指定
                X_eval_cv = self.test.drop(columns=drop_columns)

                pred = model.predict(X_eval_cv)
                preds.append(pred)
            
            pred_dict[target] = preds

        for target in targets:
            preds = pred_dict[target]
            for i, pred in enumerate(preds):
                self.test[f"{target}_pred_{i}"] = pred

            self.test[target] = self.test[[f"{target}_pred_{fold}" for fold in range(CFG.n_splits)]].mean(axis=1)

        self.test[["student_id", "content", "wording"]].to_csv("submission.csv", index=False)




def main():

    runner = Runner()
    runner.run()

    return runner

