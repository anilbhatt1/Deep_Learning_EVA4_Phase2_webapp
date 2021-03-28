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

min_len = 4             # We are giving min_len = 4 bcoz. ie largest Maxpool filter we used in model
nlp = en_core_web_sm.load()
qn_dict = {0: 'about Entities', 1:'about Humans', 2: 'Descriptive', 3: 'Numerical', 4: 'about Location', 5: 'about Abbreviations'}

def analyse(sentence):
     model_path = './conv_multiqn_analysis_CPU.pt'
     vocab_path = './conv_multiqn_analysis_vocab.pkl'

     with st.spinner('Classifying the question...'):

         if sentence is not None:
             if not os.path.exists(model_path):
                 gdd.download_file_from_google_drive(file_id='1-3RLT_sWfCQs0DwY_Q6MKohQGZva8Tj6', dest_path=model_path, unzip=False)
             if not os.path.exists(vocab_path):
                 gdd.download_file_from_google_drive(file_id='1-167zznJRK4SJBx3K8RM1YBdhUhLLBbn', dest_path=vocab_path, unzip=False)

             vocab_cloud_path = Path(f"./conv_multiqn_analysis_vocab.pkl")

             with vocab_cloud_path.open("rb") as vocab_file:
                 vocab = pickle.load(vocab_file)

             txt = vocab["txt.vocab"]
             lbl = vocab["lbl.vocab"]

             model = torch.jit.load(model_path)

             tokenized = [tok.text for tok in nlp.tokenizer(sentence)]   # tokenizing the sentence
             if len(tokenized) < min_len:
                 tokenized += ['<pad>'] * (min_len - len(tokenized))  # Padding tokens to match it with min_len 
             indexed = [txt.stoi[t] for t in tokenized]  # indexes the tokens by converting them into their integer representation from our vocabulary
             tensor  = torch.LongTensor(indexed)  # converts 'indexed' which is a Python list into a PyTorch tensor
             tensor = tensor.unsqueeze(1)         # adding batch dimension to feed it to GPU
             with torch.no_grad():
                 prediction    = model(tensor) 
             max_preds = prediction.argmax(dim=1).item()

             st.write('### This question is ', qn_dict[max_preds])
