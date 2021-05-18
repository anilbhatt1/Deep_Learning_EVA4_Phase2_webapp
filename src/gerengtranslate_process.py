import streamlit as st
import torch
import torchtext
import os
import spacy
import pickle
import dill
from pathlib import Path
import en_core_web_sm
import de_core_news_sm

from spacy.lang.en import English
from spacy.lang.de import German

import sys
from google_drive_downloader import GoogleDriveDownloader as gdd

min_len = 5             # We are giving min_len = 5 bcoz. ie largest Maxpool filter we used in model
nlp = en_core_web_sm.load()

def translate(sentence):
     model_path = './German_Eng_translate_CPU.pt'
     vocab_path = './german_english_vocab.pkl'

     with st.spinner('Analyzing the sentiment...'):

         if sentence is not None:
             if not os.path.exists(model_path):
                 gdd.download_file_from_google_drive(file_id='10oPjInWl0kH8kITq6HtZUgdxjf1TS0-e', dest_path=model_path, unzip=False)
             if not os.path.exists(vocab_path):
                 gdd.download_file_from_google_drive(file_id='1G4HuvGrgUMAUB8Qp_bqoLaKQNzA_yZbt', dest_path=vocab_path, unzip=False)

             vocab_cloud_path = Path(f"./german_english_vocab.pkl")

             with vocab_cloud_path.open("rb") as vocab_file:
                 vocab = dill.load(vocab_file)

