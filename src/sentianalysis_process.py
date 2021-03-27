import streamlit as st
import torch
import torchtext
import os
import spacy
import pickle
from pathlib import Path
import en_core_web_sm

import sys
from google_drive_downloader import GoogleDriveDownloader as gdd

min_len = 5             # We are giving min_len = 5 bcoz. ie largest Maxpool filter we used in model
nlp = en_core_web_sm.load()

def analyse(sentence):
     model_path = './Conv_Senti_Analysis_CPU.pt'
     vocab_path = './conv_senti_analysis_vocab.pkl'

     with st.spinner('Analyzing the sentiment...'):

         if sentence is not None:
             if not os.path.exists(model_path):
                 gdd.download_file_from_google_drive(file_id='1-2SHUf-qQ7Ucq2k_iErWRGB8onSzfmer', dest_path=model_path, unzip=False)
             if not os.path.exists(vocab_path):
                 gdd.download_file_from_google_drive(file_id='1-1XGx5dNqlAyVeq8XOSCPBOumwQPlJYm', dest_path=vocab_path, unzip=False)

             vocab_cloud_path = Path(f"./conv_senti_analysis_vocab.pkl")

             with vocab_cloud_path.open("rb") as vocab_file:
                 vocab = pickle.load(vocab_file)

             txt = vocab["txt.vocab"]
             lbl = vocab["lbl.vocab"]

             model = torch.jit.load(model_path)

             tokenized = [tok.text for tok in nlp.tokenizer(sentence)]   # tokenizing the sentence
             if len(tokenized) < min_len:
                 tokenized += ['<pad>'] * (min_len - len(tokenized))  # Padding tokens to match it with min_len 
             indexed = [txt.stoi[t] for t in tokenized]  # indexes the tokens by converting them into their integer representation from our vocabulary
             # "pathetic movie" --> indexed representation will be [1316, 22, 1, 1, 1] , 1316 ->pathetic , 22-> movie, 1 -> <pad> 
             tensor  = torch.LongTensor(indexed)  # converts 'indexed' which is a Python list into a PyTorch tensor
             tensor = tensor.unsqueeze(0)         # adding batch dimension to feed it to GPU
             with torch.no_grad():
                 prediction    = torch.sigmoid(model(tensor)) # Using sigmoid to keep the predictions between 0 & 1
             sentiment = prediction.item()

             if sentiment <= 0.35:               
                st.write('### Review is Negative, sentiment-score is        ', round(sentiment,3))
             elif sentiment > 0.35 and sentiment <= 0.6:
                st.write('### Review is Neutral, sentiment-score is      ', round(sentiment,3))
             else:
                st.write('### Review is Positive, sentiment-score is        ', round(sentiment,3))
