from io import BytesIO
from typing import List, Tuple, Dict

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import streamlit as st
import requests

from PIL import Image
from torchvision.transforms import Compose

Mean: List = [0.48043839, 0.44820218, 0.39760034]
STD: List = [0.27698959, 0.26908774, 0.28216029]
CLASS_NAMES = ['Flying Birds', 'Large QuadCopters', 'Small QuadCopters', 'Winged Drones']

TRANSFORMS: Compose = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=Mean,std=STD),
    ])

def mobilenet_classify(model: nn.Module, image: BytesIO):
    img: Image  = Image.open(image).convert('RGB')
    img: Tensor = TRANSFORMS(img)
    img.unsqueeze_(0)

    model.eval()
    with torch.no_grad():
        labels_pred: Tensor = model(img)
        labels_pred_max  = labels_pred.argmax(dim =1, keepdim = True)

    labels_pred_max = labels_pred_max.cpu().numpy()
    return labels_pred_max

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        print('get_confirm_token')
        for key, value in response.cookies.items():
            print('get_confirm_token inside for')
            if key.startswith('download_warning'):
                print('value:',value)
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        print('save_response_content')
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                print('save_response_content inside for')
                if chunk: # filter out keep-alive new chunks
                    print('inside if chunk ')
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    print('url:',URL)
    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    print('token value:', token)
    if token:
        print('inside token if, id:' , id)
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)
