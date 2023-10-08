from typing import List, Optional
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
import time
from textblob import TextBlob
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from oauth2client.service_account import ServiceAccountCredentials
from tqdm import tqdm

import nltk
from nltk import pos_tag
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
    several_layer: bool=False
    cls_pooling: bool=False
    additional_features: bool=True
    automodel: bool = False

class RCFG:
    run_name: str = 'run'
    commit_hash: str =""
    debug: bool =True
    debug_size: int =10
    train: bool = True
    predict: bool = True
    debug_infer: bool = False
    base_model_dir: str = "/kaggle/input/debertav3base"
    output_path: str = ""
    model_dir: str = "." # "/kaggle/commonlit-models"
    data_dir: str = "/kaggle/input/commonlit-evaluate-student-summaries/"
    use_aug_data: bool = False
    aug_data_dir: str = "/kaggle/input/commonlit-aug-data/"
    gensim_bin_model_path: str = "/kaggle/input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin"
    metadata_path: str = "/kaggle/input/commonlit-text-metadata/prompt_grade_simple.csv"
    lgbm_model_dir: Optional[str] = None
    aug_data_list: [
        "back_translation_Hel_fr"
    ]
    save_to_sheet: str = True
    sheet_json_key: str = '/kaggle/input/ktokunagautils/ktokunaga-4094cf694f5c.json'
    sheet_key: str = '1LhmdqSXborxoP1Pwb1ly-UO_DTfGSfXDN25ZS5MkvHI'
    kaggle_dataset_title: str = "commonlit-models"
    input_cols: List[str] = ["prompt_title", "prompt_question", "fixed_summary_text"]
    lgbm_repeat_cnt: int = 5
    random_seed_list: List[int] = [42, 83, 120, 2203, 1023]
    lgbm_params = {
        'boosting_type': 'gbdt',
        'random_state': 42,
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.02,
        'max_depth': 4,
        'lambda_l1': 0.0,
        'lambda_l2': 0.011
     }
    additional_features = [
        "summary_length", 
        "splling_err_num",
        "word_overlap_count",
        "bigram_overlap_count",
        "bigram_overlap_ratio",
        "trigram_overlap_count",
        "trigram_overlap_ratio",
        "quotes_count",
        'num_unq_words', # ここから追加分
        'num_chars',
        'avg_word_length', 
        'comma_count', 
        'semicolon_count',
        'flesch_reading_ease', 
        'word_count', 
        'sentence_length',
        'vocabulary_richness', 
        'gunning_fog', 
        'flesch_kincaid_grade_level',
        'count_difficult_words',
        'keyword_density',
        'jaccard_similarity',
        'text_similarity',
        'pos_mean',
        'punctuation_sum',
        'grade', # ここからメタデータ
        'lexile',
        'lexile_scaled',
        'is_prose',
        'author_type',
        'author_frequency'
    ]
    report_to: str = "wandb" # noneT
    on_kaggle: bool = True
    use_preprocessed_dataset: bool = True
    preprocessed_dataset_path: str = ""
    use_train_data_file: str = "train_preprocessed.csv"
    use_test_data_file: str = "test_preprocessed.csv"
    use_lgbm: bool = False
    scaling: bool = False
    join_metadata: bool = True
    wandb_api_key: Optional[str] = None

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
        gs = gspread.authorize(credentials)
        self.worksheet = gs.open_by_key(sheet_key)
    

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
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
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
        
        del gensim_model
        gc.collect()

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
        
        prompt_id = row['prompt_id']
        prompt_words = self.prompt_token[prompt_id]
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
        prompt_id = row['prompt_id']
        original_tokens = self.prompt_token[prompt_id]
        summary_tokens = row['summary_tokens']

        # Generate n-grams for the original text and summary
        original_ngrams = set(self.ngrams(original_tokens, n))
        summary_ngrams = set(self.ngrams(summary_tokens, n))

        # Calculate the number of common n-grams
        common_ngrams = original_ngrams.intersection(summary_ngrams)

        return len(common_ngrams)
    
    def quotes_count(self, row):
        summary = row['text']
        prompt_id = row['prompt_id']
        text = self.prompt_text[prompt_id]
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
        for tokens in series:
            prompt_words.update(tokens)
        self.prompt_words_in_order = [item[0] for item in prompt_words.most_common()] 

        self.all_words_rank = {}
        for i,word in enumerate(self.prompt_words_in_order + self.gensim_words):
            if word in self.all_words_rank:
                continue
            self.all_words_rank[word] = i
    
    def create_prompt_dictionary(self, prompts):

        self.prompt_text = dict()
        self.prompt_token = dict()
        
        for id, text, token in zip(prompts['prompt_id'], prompts['prompt_text'], prompts['prompt_tokens']):
            self.prompt_text[id] = text
            self.prompt_token[id] = token

    def calculate_unique_words(self,text):
        unique_words = set(text.split())
        return len(unique_words)
    
    def calculate_keyword_density(self,row):
        prompt_id = row['prompt_id']
        keywords = set(self.prompt_text[prompt_id].split())
        text_words = row['text'].split()
        keyword_count = sum(1 for word in text_words if word in keywords)
        return keyword_count / len(text_words)
    
    def count_syllables(self,word):

        VOWEL_RUNS = re.compile("[aeiouy]+", flags=re.I)
        EXCEPTIONS = re.compile("[^aeiou]e[sd]?$|" + "[^e]ely$", flags=re.I)
        ADDITIONAL = re.compile("[^aeioulr][lr]e[sd]?$|[csgz]es$|[td]ed$|"+ ".y[aeiou]|ia(?!n$)|eo|ism$|[^aeiou]ire$|[^gq]ua",flags=re.I)

        vowel_runs = len(VOWEL_RUNS.findall(word))
        exceptions = len(EXCEPTIONS.findall(word))
        additional = len(ADDITIONAL.findall(word))
            
        return max(1, vowel_runs - exceptions + additional)
    

    def flesch_reading_ease_manual(self,text):
        total_sentences = len(TextBlob(text).sentences)
        total_words = len(TextBlob(text).words)
        total_syllables = sum(self.count_syllables(word) for word in TextBlob(text).words)

        if total_sentences == 0 or total_words == 0:
            return 0

        flesch_score = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
        return flesch_score
    
    def flesch_kincaid_grade_level(self, text):
        total_sentences = len(TextBlob(text).sentences)
        total_words = len(TextBlob(text).words)
        total_syllables = sum(self.count_syllables(word) for word in TextBlob(text).words)

        if total_sentences == 0 or total_words == 0:
            return 0

        fk_grade = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
        return fk_grade

    def gunning_fog(self, text):
        total_sentences = len(TextBlob(text).sentences)
        total_words = len(TextBlob(text).words)
        complex_words = sum(1 for word in TextBlob(text).words if self.count_syllables(word) > 2)

        if total_sentences == 0 or total_words == 0:
            return 0

        fog_index = 0.4 * ((total_words / total_sentences) + 100 * (complex_words / total_words))
        return fog_index
    
    def count_difficult_words(self, text, syllable_threshold=3):
        words = TextBlob(text).words
        difficult_words_count = sum(1 for word in words if self.count_syllables(word) >= syllable_threshold)
        return difficult_words_count
    
    def calculate_text_similarity(self, row):
        vectorizer = TfidfVectorizer()
        prompt_id = row['prompt_id']
        tfidf_matrix = vectorizer.fit_transform([self.prompt_text[prompt_id], row['text']])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]).flatten()[0]

    def calculate_pos_ratios(self , text):
        pos_tags = pos_tag(nltk.word_tokenize(text))
        pos_counts = Counter(tag for word, tag in pos_tags)
        total_words = len(pos_tags)
        ratios = {tag: count / total_words for tag, count in pos_counts.items()}
        return ratios
    
    def calculate_punctuation_ratios(self,text):
        total_chars = len(text)
        punctuation_counts = Counter(char for char in text if char in '.,!?;:"()[]{}')
        ratios = {char: count / total_chars for char, count in punctuation_counts.items()}
        return ratios
    
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

        # prompts['gunning_fog_prompt'] = prompts['prompt_text'].apply(self.gunning_fog)
        # prompts['flesch_kincaid_grade_level_prompt'] = prompts['prompt_text'].apply(self.flesch_kincaid_grade_level)
        # prompts['flesch_reading_ease_prompt'] = prompts['prompt_text'].apply(self.flesch_reading_ease_manual)
        
        # fix misspelling
        summaries["fixed_summary_text"] = summaries["summary_tokens"].progress_apply(
            lambda x: self.fix_text(x)
        )
        
        # count misspelling
        summaries["splling_err_num"] = summaries["text"].progress_apply(self.spelling)

        self.create_prompt_dictionary(prompts)
        prompts = prompts.drop(['prompt_text', 'prompt_tokens'], axis=1)
        gc.collect()
        
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

        # 後から追加したもの
        input_df['num_unq_words']=[len(list(set(x.lower().split(' ')))) for x in input_df.text]
        input_df['num_chars']= [len(x) for x in input_df.text]
        input_df['avg_word_length'] = input_df['text'].apply(lambda x: np.mean([len(word) for word in x.split()]))
        input_df['comma_count'] = input_df['text'].apply(lambda x: x.count(','))
        input_df['semicolon_count'] = input_df['text'].apply(lambda x: x.count(';'))
        input_df['flesch_reading_ease'] = input_df['text'].apply(self.flesch_reading_ease_manual)
        input_df['word_count'] = input_df['text'].apply(lambda x: len(x.split()))
        input_df['sentence_length'] = input_df['text'].apply(lambda x: len(x.split('.')))
        input_df['vocabulary_richness'] = input_df['text'].apply(lambda x: len(set(x.split())))
        input_df['gunning_fog'] = input_df['text'].apply(self.gunning_fog)
        input_df['flesch_kincaid_grade_level'] = input_df['text'].apply(self.flesch_kincaid_grade_level)
        input_df['count_difficult_words'] = input_df['text'].apply(self.count_difficult_words)
        input_df['keyword_density'] = input_df.apply(self.calculate_keyword_density, axis=1)
        input_df['jaccard_similarity'] = input_df.apply(
            lambda row: len(set(word_tokenize(self.prompt_text[row['prompt_id']])) & set(word_tokenize(row['text']))) / len(set(word_tokenize(self.prompt_text[row['prompt_id']])) | set(word_tokenize(row['text']))), axis=1
        )
        input_df['text_similarity'] = input_df.progress_apply(self.calculate_text_similarity, axis=1)
        input_df['pos_ratios'] = input_df['text'].apply(self.calculate_pos_ratios)
        input_df['pos_mean'] = input_df['pos_ratios'].apply(lambda x: np.mean(list(x.values())))
        input_df['punctuation_ratios'] = input_df['text'].apply(self.calculate_punctuation_ratios)
        input_df['punctuation_sum'] = input_df['punctuation_ratios'].apply(lambda x: np.sum(list(x.values())))

        df_features = input_df[RCFG.additional_features].copy()

        return input_df.drop(columns=["summary_tokens", "prompt_length", "pos_ratios", "punctuation_ratios"])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

class MCRMSELoss(nn.Module):
    def __init__(self, num_scored=1):
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
            additional_features_dim, 
            n_freeze = 0, 
            dropout=0.1
        ):
        super(CustomTransformersModel, self).__init__()
        self.base_model = base_model
        self.additional_features_dim = additional_features_dim
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(base_model.config.hidden_size, 1)

        # freezing embeddings layer
        if n_freeze:
            self.base_model.embeddings.requires_grad_(False)
        
            #freezing the initial N layers
            for i in range(0, n_freeze, 1):
                for n,p in self.base_model.encoder.layer[i].named_parameters():
                    p.requires_grad = False

        self.creterion = MCRMSELoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

        if CFG.mean_pooling:
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            base_model_output = sum_embeddings / sum_mask
        else:
            base_model_output = outputs[0][:, 0, :]
        
        logits = self.classifier(self.dropout(base_model_output))

        if labels is not None:
            loss = self.creterion(logits, labels)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}

# several layer
class CustomTransformersModelV2(nn.Module):
    def __init__(
            self, 
            base_model, 
            additional_features_dim, 
            n_freeze = 0, 
            dropout=0.1
        ):
        super(CustomTransformersModelV2, self).__init__()
        self.base_model = base_model
        self.additional_features_dim = additional_features_dim
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(base_model.config.hidden_size*4, 1)

        # freezing embeddings layer
        if n_freeze:
            self.base_model.embeddings.requires_grad_(False)
        
            #freezing the initial N layers
            for i in range(0, n_freeze, 1):
                for n,p in self.base_model.encoder.layer[i].named_parameters():
                    p.requires_grad = False

        self.creterion = MCRMSELoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        base_model_output = torch.cat(outputs.hidden_states[-4:], 2)[:, 0, :]
        
        logits = self.classifier(self.dropout(base_model_output))

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
        target: str,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        max_length: int,
        logger
    ):


        self.input_col = "input"        
        self.input_text_cols = inputs 
        self.target = target
        self.additional_feature_cols = RCFG.additional_features

        self.model_name = model_name
        self.model_dir = model_dir
        self.max_length = max_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(f"{RCFG.base_model_dir}")
        self.model_config = AutoConfig.from_pretrained(f"{RCFG.base_model_dir}")
        
        self.model_config.update({
            "hidden_dropout_prob": hidden_dropout_prob,
            "attention_probs_dropout_prob": attention_probs_dropout_prob,
            "num_labels": 1,
            "problem_type": "regression"
        })

        if CFG.several_layer:
            self.model_config.update({
                "output_hidden_states": True
            })
        
        seed_everything(seed=42)

        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer
        )
        self.logger=logger

    def concatenate_with_sep_token(self, row):
        sep = " " + self.tokenizer.sep_token + " "        
        return sep.join(row[self.input_text_cols])

    def tokenize_function(self, examples: pd.DataFrame):
        labels = [examples[self.target]]
        tokenized = self.tokenizer(examples[self.input_col],
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length)
        return {
            **tokenized,
            "labels": labels
        }
    
    def tokenize_function_test(self, examples: pd.DataFrame):
        tokenized = self.tokenizer(examples[self.input_col],
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length)
        return tokenized
    
    def get_custom_model(self, ):

        model_content = AutoModel.from_pretrained(
            f"{RCFG.base_model_dir}", 
            config=self.model_config
        )

        if CFG.several_layer:
            self.logger.info('Use CustomTransformerModelV3 with last 4 transformer layers.')
            custom_model = CustomTransformersModelV2(
                model_content,
                additional_features_dim=len(self.additional_feature_cols),
                n_freeze=CFG.n_freeze
            )
        elif CFG.automodel:
            self.logger.info('Use AutoModelForSequenceClassification.')
            custom_model = AutoModelForSequenceClassification.from_pretrained(
                f"{RCFG.base_model_dir}", 
                config=self.model_config
            )
        else:
            if CFG.mean_pooling:
                self.logger.info('Use CustomTransformerModel with mean_pooling.')
            else:
                self.logger.info('Use CustomTransformerModel with CLS token.')
            custom_model = CustomTransformersModel(
                model_content,
                additional_features_dim=len(self.additional_feature_cols),
                n_freeze=CFG.n_freeze
            )

        return custom_model

    def train(
        self, 
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
        
        train_df = train_df[[self.input_col] + [self.target]]
        valid_df = valid_df[[self.input_col] + [self.target]]
        
        custom_model = self.get_custom_model()

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
            metric_for_best_model="rmse",
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
            compute_metrics=compute_metrics,
            data_collator=self.data_collator
        )

        trainer.train()
        
        self.tokenizer.save_pretrained(self.model_dir)
        torch.save(custom_model.state_dict(), os.path.join(self.model_dir, "model_weight.pth"))

        custom_model.cpu()
        del custom_model
        gc.collect()
        torch.cuda.empty_cache()
    
        
    def predict(
        self, 
        test_df: pd.DataFrame,
        batch_size: int
    ):
        """predict content score"""
        
        test_df[self.input_col] = test_df.apply(self.concatenate_with_sep_token, axis=1)
        test_df = test_df[[self.input_col]]

        test_dataset = Dataset.from_pandas(test_df, preserve_index=False) 
        test_tokenized_dataset = test_dataset.map(self.tokenize_function_test, batched=False)

        custom_model = self.get_custom_model()
        custom_model.load_state_dict(torch.load(os.path.join(self.model_dir, "model_weight.pth")))
        custom_model.eval()
        
        test_args = TrainingArguments(
            output_dir=".",
            do_train = False,
            do_predict = True,
            per_device_eval_batch_size=12,
            dataloader_drop_last = False,
            fp16=True,
            save_strategy="no"
        )

        # init trainer
        infer_content = Trainer(
            model = custom_model, 
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            args = test_args
        )

        preds = infer_content.predict(test_tokenized_dataset)[0]
        pred_df = pd.DataFrame(preds, columns=[f"{self.target}_pred"])
        
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

    for target in targets:
        logger.info(f'target: {target}.')

        for fold in range(CFG.n_splits):
            logger.info(f"fold {fold}:")
            
            train_data = train_df[train_df["fold"] != fold]
            valid_data = train_df[train_df["fold"] == fold]

            if RCFG.use_aug_data and target == 'content': 
                logger.info('Augment data by back translation.')
                train_aug_data = df_augtrain[df_augtrain["fold"] != fold]
                train_data = pd.concat([train_data, train_aug_data])
            
            fold_model_dir = f'{model_dir}/{target}/fold_{fold}'
            csr = ScoreRegressor(
                model_name=model_name,
                target=target,
                inputs= inputs,
                model_dir = fold_model_dir, 
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                max_length=max_length,
                logger=logger
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
    
    for target in targets:
        logger.info(f'target: {target}.')

        for fold in range(CFG.n_splits):
            logger.info(f"fold {fold}:")
            
            valid_data = train_df[train_df["fold"] == fold]
            
            model_dir =  f"{RCFG.model_dir}/{model_name}/{target}/fold_{fold}"
            
            csr = ScoreRegressor(
                model_name=model_name,
                target=target,
                inputs= inputs,
                model_dir = model_dir,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                max_length=max_length,
                logger=logger
            )
            
            pred_df = csr.predict(
                test_df=valid_data, 
                batch_size=batch_size
            )

            train_df.loc[valid_data.index, f"{target}_pred"] = pred_df[f"{target}_pred"].values
                
    for target_pred in ["content_pred", "wording_pred"]:
        if target_pred not in columns: columns.append(target_pred)
    
    return train_df[columns]


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

    for target in targets:
        logger.info(f'target: {target}.')

        for fold in range(CFG.n_splits):
            logger.info(f"fold {fold}:")
            
            model_dir =  f"{RCFG.model_dir}/{model_name}/{target}/fold_{fold}"
            logger.info(f'prediction model dir: {model_dir}.')

            csr = ScoreRegressor(
                model_name=model_name,
                target=target,
                inputs= inputs,
                model_dir = model_dir, 
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                max_length=max_length,
                logger=logger
            )

            pred_df = csr.predict(
                test_df=test_df, 
                batch_size=batch_size
            )

            del csr
            gc.collect()
            torch.cuda.empty_cache()

            test_df[f"{target}_pred_{fold}"] = pred_df[f"{target}_pred"].values

        test_df[f"{target}_pred"] = test_df[[f"{target}_pred_{fold}" for fold in range(CFG.n_splits)]].mean(axis=1)

    for target_pred in ["content_pred", "wording_pred"]:
        if target_pred not in columns: columns.append(target_pred)
    
    return test_df[columns]


def join_metadata(df1, df2, df1_title_col, df2_title_col, grade_col):
    # Copy dataframes to avoid modifying the originals
    df1 = df1.copy()
    df2 = df2.copy()

    # Preprocess titles
    df1[df1_title_col] = df1[df1_title_col].str.replace('"', '').str.strip()
    df2[df2_title_col] = df2[df2_title_col].str.replace('"', '').str.strip()

    # Remove duplicate grades
    df2 = df2.drop_duplicates(subset=df2_title_col, keep='first')

    # Join dataframes
    merged_df = df1.merge(df2, how='left', left_on=df1_title_col, right_on=df2_title_col)
    

    # Postprocess grades
    merged_df[grade_col] = merged_df[grade_col].fillna(0)
    merged_df[grade_col] = merged_df[grade_col].astype(int).astype('category')

 
    return merged_df


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
        self.lgbm_columns = RCFG.additional_features + ["content_pred", "wording_pred"]

        self.data_to_write = []

        if RCFG.lgbm_model_dir is None:
            RCFG.lgbm_model_dir = RCFG.model_dir

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

        if RCFG.train:
            self.prompts_train = pd.read_csv(RCFG.data_dir + "prompts_train.csv")
            self.summaries_train = pd.read_csv(RCFG.data_dir + "summaries_train.csv")

            if RCFG.join_metadata:
                self.logger.info('Use prompts metadata.')
                prompt_grade = pd.read_csv(RCFG.metadata_path)
                self.prompts_train = join_metadata(self.prompts_train, prompt_grade, 'prompt_title', 'title', 'grade')
        
        if RCFG.predict:
            self.prompts_test = pd.read_csv(RCFG.data_dir + "prompts_test.csv")
            self.summaries_test = pd.read_csv(RCFG.data_dir + "summaries_test.csv")

            if RCFG.join_metadata:
                prompt_grade = pd.read_csv(RCFG.metadata_path)
                self.prompts_test = join_metadata(self.prompts_test, prompt_grade, 'prompt_title', 'title', 'grade')

        self.sample_submission = pd.read_csv(RCFG.data_dir + "sample_submission.csv")

        if RCFG.debug:
            self.logger.info('Debug mode. Reduce train data.')
            self.summaries_train = self.summaries_train.head(RCFG.debug_size) # for dev mode

        if RCFG.debug_infer:
            self.prompts_train = pd.read_csv(RCFG.data_dir + "prompts_train.csv")
            self.summaries_train = pd.read_csv(RCFG.data_dir + "summaries_train.csv")
            self.prompts_test = self.prompts_train.copy()
            self.summaries_test = pd.concat([self.summaries_train, self.summaries_train, self.summaries_train])[:17000]
            
        
        self.augtrain = None
        if RCFG.use_aug_data:
            
            self.augtrain = pd.read_csv(RCFG.aug_data_dir + "back_translation_all.csv")
            self.augtrain = self.augtrain[self.augtrain['lang'].isin(RCFG.aug_data_list)].drop(['lang'], axis=1)
            self.augtrain.columns = ['student_id', 'fixed_summary_text']

    def preprocess(self):

        if RCFG.use_preprocessed_dataset:
            self.logger.info(f'Use exsisting files.')
            nrows = RCFG.debug_size if RCFG.debug else None

            if RCFG.train:
                self.train = pd.read_csv(f"{RCFG.preprocessed_dataset_path}/{RCFG.use_train_data_file}", nrows=nrows)
                if 'grade' in self.train.columns:
                    self.train['grade'] = self.train['grade'].astype('category')
            if RCFG.predict:
                self.test = pd.read_csv(f"{RCFG.preprocessed_dataset_path}/{RCFG.use_test_data_file}")
                if 'grade' in self.test.columns:
                    self.test['grade'] = self.test['grade'].astype('category')
                
            return None

        self.logger.info('Start Preprocess.')
        preprocessor = Preprocessor(model_name=CFG.model_name)

        if RCFG.train:
            self.logger.info('Preprocess train data.')
            self.train = preprocessor.run(self.prompts_train, self.summaries_train, mode="train")
        
        if RCFG.predict:
            self.logger.info('Preprocess test data.')
            self.test = preprocessor.run(self.prompts_test, self.summaries_test, mode="test")

            del self.prompts_test
            del self.summaries_test
            gc.collect()

            self.test.to_csv('test_preprocessed.csv', index=False)
        
        del preprocessor
        gc.collect()

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

        if RCFG.train:
            
            torch.cuda.empty_cache()
            print_gpu_utilization(self.logger) # 2, 7117　(2, 6137)
            self.logger.info(f'Start training by fold.')
            
            train_by_fold(
                logger=self.logger,
                train_df=self.train,
                model_name=CFG.model_name,
                targets=self.targets,
                inputs=RCFG.input_cols,
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
            
            print_gpu_utilization(self.logger) 
            self.logger.info(f'Start creating oof prediction.')
            self.train = validate(
                logger=self.logger,
                train_df=self.train,
                targets=self.targets,
                inputs=RCFG.input_cols,
                batch_size=CFG.batch_size,
                model_name=CFG.model_name,
                hidden_dropout_prob=CFG.hidden_dropout_prob,
                attention_probs_dropout_prob=CFG.attention_probs_dropout_prob,
                max_length=CFG.max_length
            )

            # set validate result
            rmses = []
            for target in self.targets:
                rmse = mean_squared_error(self.train[target], self.train[f"{target}_pred"], squared=False)
                self.logger.info(f"cv {target} rmse: {rmse}")
                self.data_to_write.append(rmse)
                rmses.append(rmse)
            self.data_to_write.append(np.mean(rmses))
        
        if RCFG.predict:
            
            # time.sleep(3600)
            self.test['fixed_summary_text'] = self.test['fixed_summary_text'].str[:1000]
            print_gpu_utilization(self.logger)
            self.logger.info(f'Start Predicting.')
            self.test = predict(
                logger=self.logger,
                test_df=self.test,
                targets=self.targets,
                inputs = RCFG.input_cols,
                batch_size=CFG.batch_size,
                model_name=CFG.model_name,
                hidden_dropout_prob=CFG.hidden_dropout_prob,
                attention_probs_dropout_prob=CFG.attention_probs_dropout_prob,
                max_length=CFG.max_length
            )

            id = RCFG.model_dir[-2:]
            self.test[['student_id', 'content_pred', 'wording_pred']].to_csv(f'{id}_tmp.csv', index=False) 
            print_gpu_utilization(self.logger)
        
        if RCFG.train:
            self.train.to_csv(f'{RCFG.model_dir}/train_processed_pred_{RCFG.commit_hash}.csv', index=False)
        
        if RCFG.report_to == 'wandb':
            
            import wandb
            wandb.finish()

    def run_lgbm(self):

        if not RCFG.use_lgbm or not RCFG.train:
            return None
        
        self.model_dict = {}

        for target in self.targets:
            self.logger.info(f'Start training LGBM model: {target}')

            # repeat 5 times chaning random seed
            for i in range(RCFG.lgbm_repeat_cnt):
                RCFG.lgbm_params['random_state'] = RCFG.random_seed_list[i]

                models = []
                for fold in range(CFG.n_splits):

                    X_train_cv = self.train[self.train["fold"] != fold][self.lgbm_columns]
                    y_train_cv = self.train[self.train["fold"] != fold][target]

                    X_eval_cv = self.train[self.train["fold"] == fold][self.lgbm_columns]
                    y_eval_cv = self.train[self.train["fold"] == fold][target]

                    dtrain = lgb.Dataset(X_train_cv, label=y_train_cv)
                    dval = lgb.Dataset(X_eval_cv, label=y_eval_cv)

                    evaluation_results = {}
                    model = lgb.train(
                        RCFG.lgbm_params,
                        num_boost_round=10000,
                        valid_names=['train', 'valid'],
                        train_set=dtrain,
                        valid_sets=dval,
                        verbose_eval=False,
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=30, verbose=False),
                            lgb.log_evaluation(-1),
                            lgb.callback.record_evaluation(evaluation_results)
                        ],
                    )
                    models.append(model)
            
                self.model_dict[f'{target}_{i}'] = models

        # cv
        rmses = []

        for target in self.targets:
            
            preds = []
            for i in range(RCFG.lgbm_repeat_cnt):
                models = self.model_dict[f'{target}_{i}']

                preds_tmp = []
                trues = []
                for fold, model in enumerate(models):
                    # ilocで取り出す行を指定
                    X_eval_cv = self.train[self.train["fold"] == fold][self.lgbm_columns]
                    y_eval_cv = self.train[self.train["fold"] == fold][target]

                    pred_tmp = model.predict(X_eval_cv)

                    trues.extend(y_eval_cv)
                    preds_tmp.extend(pred_tmp)
                
                preds.append(preds_tmp)
            
            preds = np.array(preds_tmp).mean(axis=0)
                
            rmse = np.sqrt(mean_squared_error(trues, preds))
            self.logger.info(f"{target}_rmse : {rmse}")
            self.data_to_write.append(rmse)
            rmses = rmses + [rmse]

        mcrmse = sum(rmses) / len(rmses)
        self.logger.info(f"mcrmse : {mcrmse}")
        self.data_to_write.append(mcrmse)


        # delete old model files
        model_dir = f'{RCFG.lgbm_model_dir}/gbtmodel'
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)

        save_model_path = f'{model_dir}/model_dict.pkl'
        self.logger.info(f'save LGBM model: {save_model_path}')
        with open(save_model_path, 'wb') as f:
            pickle.dump(self.model_dict, f)


    def create_prediction(self, filename="submission.csv"):

        if not RCFG.predict:
            return None
        
        if not RCFG.use_lgbm:
        
            self.logger.info('Start creating submission data without LGBM.')
            df_output = self.test[["student_id", "content_pred", "wording_pred"]]
            df_output = df_output.rename(columns={"content_pred": "content", "wording_pred": "wording"})

            df_output.to_csv("submission.csv", index=False)
            return None


        self.logger.info('Start creating submission data using LGBM.')
        with open(f'{RCFG.lgbm_model_dir}/gbtmodel/model_dict.pkl', 'rb') as f:
            self.model_dict = pickle.load(f)

        pred_dict = {}
        for target in self.targets:
            
            preds = []
            for i in range(RCFG.lgbm_repeat_cnt):
                models = self.model_dict[f'{target}_{i}']
                
                preds_tmp = []
                for fold, model in enumerate(models):
                    X_eval_cv = self.test[self.lgbm_columns]

                    pred_tmp = model.predict(X_eval_cv)
                    preds_tmp.append(pred_tmp)

                preds.append(preds_tmp)
            
            preds = np.array(preds_tmp).mean(axis=0)
            pred_dict[target] = preds

        for target in self.targets:
            preds = pred_dict[target]
            for i, pred in enumerate(preds):
                self.test[f"{target}_pred_{i}"] = pred

            self.test[target] = self.test[[f"{target}_pred_{fold}" for fold in range(CFG.n_splits)]].mean(axis=1)

        self.test[["student_id", "content", "wording"]].to_csv(filename, index=False)


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