import streamlit as st
import torch
import torchvision
from torchvision import transforms as T
from torchvision.transforms import Compose
import os
from typing import List, Tuple, Dict
from torchvision.transforms import Compose, ToTensor, ToPILImage, Resize
import zipfile
from zipfile import ZipFile

from PIL import Image

import sys, cv2
import numpy as np
from google_drive_downloader import GoogleDriveDownloader as gdd

TRANSFORMS: Compose = T.Compose([
    T.Resize((400, 400)),
    T.ToTensor(),
    T.Lambda(lambda x: x.mul(255))
])

def generate(file):
    if file is not None:
        style = st.selectbox("Select the style", ["Mosaic", "Udnie", "Candy", "Rain_Princess", "All"])
        if st.checkbox("Generate style transferred image"):
            style_process(style, file)
    else:
        st.exception("ExceptionError('Please give an input image in valid format')")

def style_process(style, file):
    model_candy_path        = './candy_cpu_scripted.pt'
    model_udnie_path        = './udnie_cpu_scripted.pt'
    model_mosaic_path       = './mosaic_cpu_scripted.pt'
    model_rainprincess_path = './rain_princess_cpu_scripted.pt'

    with st.spinner('Generating style image...'):
        '''
         Commenting out downloading zip file as heroku app is facing time-out...

         if not os.path.exists(model_zip_path):
            gdd.download_file_from_google_drive(file_id='1RclvV5Ep2c2BYxGDxQyEHFgnkcVQG8J8', dest_path='./fast_style_transfer_models_cpu.zip', unzip=False)
            st.text('Model zip file downloaded')         

            with ZipFile('./fast_style_transfer_models_cpu.zip', mode='r') as input:
                input.extractall('.')
                st.text('Models unzipped and downloaded')
        '''
        if not os.path.exists(model_candy_path):
            gdd.download_file_from_google_drive(file_id='1zc_Y1EizuOoHd-v-dKmHJ-bEQq2HFjA2', dest_path='./candy_cpu_scripted.pt', unzip=False)
            st.text('Candy model file downloaded')

        if not os.path.exists(model_udnie_path):
            gdd.download_file_from_google_drive(file_id='13iyRJC8vMblpMUORkXQL9eKOHXU3-XIu', dest_path='./udnie_cpu_scripted.pt', unzip=False)
            st.text('Udnie model file downloaded')

        if not os.path.exists(model_mosaic_path):
            gdd.download_file_from_google_drive(file_id='1LFW1FyjC2MD_iMVBkCAeiA8eKhN-0JIw', dest_path='./mosaic_cpu_scripted.pt', unzip=False)
            st.text('Mosaic model file downloaded')

        if not os.path.exists(model_rainprincess_path):
            gdd.download_file_from_google_drive(file_id='1LtFipvF8To2bMFhfAyRLHUpeUSkOC92N', dest_path='./rain_princess_cpu_scripted.pt', unzip=False)
            st.text('Rain-Princess model file downloaded')

        pil_img     = Image.open(file).convert('RGB')
        st.text('pil_img generated')
        content_img = TRANSFORMS(pil_img)
        content_img.unsqueeze_(0)
        st.text('transformed')
        if style == 'Mosaic':
            st.text('Entering Mosaic style')
            img_m       = gen_style_img(content_img, model_mosaic_path)
            st.text('Mosaic generated')
            img_lst     = [pil_img, img_m]
            caption_lst = ['Input Img','Mosaic Style Img']
            st.text('Mosaic list complete')
        elif style == 'Udnie':
            st.text('Entering Udenie style')
            img_u       = gen_style_img(content_img, model_udnie_path)
            img_lst     = [pil_img, img_u]
            caption_lst = ['Input Img','Udnie Style Img']  
        elif style == 'Candy':
            img_c       = gen_style_img(content_img, model_candy_path)
            img_lst     = [pil_img, img_c]
            caption_lst = ['Input Img','Candy Style Img']
        elif style == 'Rain_Princess':
            img_r       = gen_style_img(content_img, model_rainprincess_path)
            img_lst     = [pil_img, img_r]
            caption_lst = ['Input Img','Rain-Princess Style Img']
        else:
            img_m       = gen_style_img(content_img, model_mosaic_path)
            img_u       = gen_style_img(content_img, model_udnie_path)
            img_c       = gen_style_img(content_img, model_candy_path)
            img_r       = gen_style_img(content_img, model_rainprincess_path)
            img_lst     = [pil_img, img_m, img_u, img_c, img_r]
            caption_lst = ['Input Img','Mosaic Style Img','Udnie Style Img','Candy Style Img','Rain-Princess Style Img'] 
        st.text('About to display') 
        st.image(img_lst,caption=caption_lst, width=250)

def gen_style_img(content_img, model_path):
         st.text('Trying to load model')
         style_mosaic = torch.jit.load(model_path)
         st.text('Model loaded')
         with torch.no_grad():
             mosaic_img  = style_mosaic(content_img)
             st.text('Generated image from model')
         mosaic_img = mosaic_img.squeeze(0).permute(1, 2, 0).clamp(0,255).numpy()
         mosaic_img = Image.fromarray(mosaic_img.astype("uint8"))
         st.text('Modified image & about to return')
         return mosaic_img
