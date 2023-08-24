from typing import List
import numpy as np 
import pandas as pd 
import warnings
import logging
import os
import random
import pickle
import shutil
import subprocess
import json
import datetime
from tqdm import tqdm
from dataclasses import dataclass, asdict

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from collections import Counter
import spacy
import re
from autocorrect import Speller
from spellchecker import SpellChecker

import nlpaug.augmenter.word as naw


@dataclass
class RunConfig():
    debug: bool =True
    debug_size: int =10
    logger_path: str = ""
    data_dir: str = "/kaggle/input/commonlit-evaluate-student-summaries/"


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
    def __init__(self, ) -> None:

        self.speller = Speller(lang='en')
        self.spellchecker = SpellChecker() 

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
            summaries:pd.DataFrame
        ) -> pd.DataFrame:
        
        prompts["prompt_tokens"] = prompts["prompt_text"].apply(
            lambda x: word_tokenize(x)
        )
        summaries["summary_tokens"] = summaries["text"].apply(
            lambda x: word_tokenize(x)
        )
        prompts["prompt_tokens"].apply(
            lambda x: self.add_spelling_dictionary(x)
        )
        summaries["fixed_summary_text"] = summaries["text"].progress_apply(
            lambda x: self.speller(x)
        )
        
        return input_df.drop(columns=["summary_tokens", "prompt_tokens"])


class Runner():

    def __init__(
        self,
    ):
    
        tqdm.pandas()
        self.logger = Logger(RunConfig.logger_path)

    def load_dataset(self):

        self.prompts_train = pd.read_csv(RunConfig.data_dir + "prompts_train.csv")
        self.prompts_test = pd.read_csv(RunConfig.data_dir + "prompts_test.csv")
        self.summaries_train = pd.read_csv(RunConfig.data_dir + "summaries_train.csv")
        self.summaries_test = pd.read_csv(RunConfig.data_dir + "summaries_test.csv")
        self.sample_submission = pd.read_csv(RunConfig.data_dir + "sample_submission.csv")

        if RunConfig.debug:
            self.logger.info('Debug mode. Reduce train data.')
            self.summaries_train = self.summaries_train.head(RunConfig.debug_size) # for dev mode


    def preprocess(self):

        preprocessor = Preprocessor()
        self.train = preprocessor.run(self.prompts_train, self.summaries_train)
        self.test = preprocessor.run(self.prompts_test, self.summaries_test)


    def translate(self):

        model_Helsinki = [
            ["Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-fr-en"],
            ["Helsinki-NLP/opus-mt-en-zh", "Helsinki-NLP/opus-mt-zh-en"],
            ["Helsinki-NLP/opus-mt-en-ru", "Helsinki-NLP/opus-mt-ru-en"],
            ["Helsinki-NLP/opus-mt-en-es", "Helsinki-NLP/opus-mt-es-en"],
            ["Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-de-en"],
            ["Helsinki-NLP/opus-mt-en-ja", "Helsinki-NLP/opus-mt-ja-en"]
        ]

        for k, models in enumerate(model_Helsinki):
            print('backtranslation: ', models)

            back_trans_aug = naw.BackTranslationAug(from_model_name=models[0], to_model_name=models[1])
            self.train[f'back_translation_{k}'] = self.train["fixed_summary_text"].apply(
                lambda x: back_trans_aug.augmente(x)
            )