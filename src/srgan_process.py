import streamlit as st
import torch
import torchvision
from torchvision import transforms as T
from torchvision.transforms import Compose
import os
from typing import List, Tuple, Dict
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize

from PIL import Image

import sys, cv2
import numpy as np
from google_drive_downloader import GoogleDriveDownloader as gdd

TRANSFORMS: Compose = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),	
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])

def generate(file):
     model_path = './SRGAN_netG_CPU_999_20210315122150.pt'

     with st.spinner('Reconstructing the input image...'):

         if file is not None:
             if not os.path.exists(model_path):
                 gdd.download_file_from_google_drive(file_id='13Q4h-AQzDqWesNbQ5CSK2rBu1JU0I2We', dest_path='./SRGAN_netG_CPU_999_20210315122150.pt', unzip=False)
                 st.text('Model downloaded')

         pil_img = Image.open(file).convert('RGB')
         sr_img = TRANSFORMS(pil_img)
         sr_img.unsqueeze_(0)

         SRGAN_Model = torch.jit.load('./SRGAN_netG_CPU_999_20210315122150.pt')
         with torch.no_grad():
             lr_img = SRGAN_Model(sr_img)
         disp_lr_img = display_transform()(lr_img.squeeze(0)).permute(1, 2, 0).numpy()
         disp_sr_img = display_transform()(sr_img.squeeze(0)).permute(1, 2, 0).numpy()
         st.image([disp_sr_img, disp_lr_img],caption=['Input Img(LR)','Reconstructed Img(HR)'], width=170)
