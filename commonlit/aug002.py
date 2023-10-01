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
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from tqdm import tqdm

import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from collections import Counter
import spacy
import re
from autocorrect import Speller
from spellchecker import SpellChecker
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import nlpaug.augmenter.word as naw

class RCFG:
    debug: bool =True
    debug_size: int =10
    logger_path: str = ""
    device: str = "cpu" # cuda
    save_path: str = "."
    use_train_data_file: str = ""
    data_dir: str = "/kaggle/input/commonlit-evaluate-student-summaries/"
    x_file_path: str = '/kaggle/commonlit/aug-data/translate_wmt21.csv'
    translation_models = [
        ["Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-fr-en"],
        ["Helsinki-NLP/opus-mt-en-zh", "Helsinki-NLP/opus-mt-zh-en"],
        ["Helsinki-NLP/opus-mt-en-ru", "Helsinki-NLP/opus-mt-ru-en"],
        ["Helsinki-NLP/opus-mt-en-es", "Helsinki-NLP/opus-mt-es-en"],
        ["Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-de-en"],
        ["Helsinki-NLP/opus-mt-en-ja", "Helsinki-NLP/opus-mt-ja-en"],
        ["facebook/wmt19-en-de", "facebook/wmt19-de-en"],
        ["facebook/wmt19-en-ru", "facebook/wmt19-ru-en"]
    ],
    langs = ['ha', 'is', 'ja', 'cs', 'ru', 'zh', 'de']


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


def print_gpu_utilization(logger):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    logger.info(f"GPU memory occupied: {info.used//1024**2} MB.")


class Runner():

    def __init__(
        self,
    ):

        tqdm.pandas()
        self.logger = Logger(RCFG.logger_path)

    def load_dataset(self):

        self.train = pd.read_csv(RCFG.use_train_data_file)

    def translate(self):

        for k, models in enumerate(RCFG.translation_models):
            print('backtranslation: ', models)
            from_model = models[0]
            to_model = models[1]
            col_suffix = f"{from_model[:3]}_{from_model[-2:]}"

            back_trans_aug = naw.BackTranslationAug(
                from_model_name=from_model, 
                to_model_name=to_model,
                device=RCFG.device)
            self.train[f'back_translation_{col_suffix}'] = self.train["fixed_summary_text"].progress_apply(
                lambda x: back_trans_aug.augment(x)[0]
            )

            # self.train[f'back_translation_{col_suffix}'] = back_trans_aug.augment(self.train["fixed_summary_text"])
    

    def save_translation_csv(self, cname="back_translation", filename='back_translation'):

        translation_columns = [c for c in self.train.columns if c.startswith(cname)]
        columns_output = ["student_id"] + translation_columns
        self.df_output = self.train[columns_output].copy()

        self.df_output = pd.melt(self.df_output, id_vars=['student_id'], value_vars=translation_columns, var_name='lang', value_name='summary_text')
        self.df_output.to_csv(f'{RCFG.save_path}/{filename}.csv', index=False)


    def translate_wmt21_to_x(self):

        def translate_lang(x, lang="de"):
            inputs = tokenizer(x, return_tensors="pt").to(device)
            generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id(lang))
            return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print_gpu_utilization(self.logger)
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt21-dense-24-wide-en-x").to(device)
        tokenizer = AutoTokenizer.from_pretrained("facebook/wmt21-dense-24-wide-en-x")
        
        for lang in RCFG.langs:
            self.logger.info(f'translation {lang}.')
            self.train[f'translate_wmt21_{lang}'] = self.train["fixed_summary_text"].progress_apply(
                translate_lang, lang=lang
            )
            
            print_gpu_utilization(self.logger)
            torch.cuda.empty_cache()
            print_gpu_utilization(self.logger)

            self.save_translation_csv(cname="translate_wmt21", filename='translate_wmt21')
    
    def translate_wmt21_to_en(self):

        df = pd.load_csv(RCFG.x_file_path)

        def translate_lang(x, lang="de"):
            tokenizer.src_lang = lang
            inputs = tokenizer(x, return_tensors="pt").to(device)
            generated_tokens = model.generate(**inputs)
            return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt21-dense-24-wide-x-en").to(device)
        tokenizer = AutoTokenizer.from_pretrained("facebook/wmt21-dense-24-wide-x-en")

        print_gpu_utilization(self.logger)
        for lang in RCFG.langs:
            tokenizer.src_lang = lang
            df_target = df[df['lang']==f'translate_wmt21_{lang}']
    
            self.train[f'back_translate_wmt21_{lang}'] = df_target['fixed_summary_text'].progress_apply(
                translate_lang, lang=lang
            )

            print_gpu_utilization(self.logger)
            torch.cuda.empty_cache()
            print_gpu_utilization(self.logger)

            self.save_translation_csv(cname="back_translate_wmt21", filename='back_translate_wmt21')

        


