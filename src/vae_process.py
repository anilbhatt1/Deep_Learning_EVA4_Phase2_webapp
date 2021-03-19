import streamlit as st
import torch
import torchvision
from torchvision import transforms as T
from torchvision.transforms import Compose
import os
from typing import List, Tuple, Dict

from PIL import Image

import sys, cv2
import numpy as np
from google_drive_downloader import GoogleDriveDownloader as gdd

Mean: List = [0.48043839, 0.44820218, 0.39760034]
STD: List = [0.27698959, 0.26908774, 0.28216029]
TRANSFORMS: Compose = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),	
    T.Normalize(mean=Mean,std=STD),
    ])


def generate(file):

     model_path = './VAE_CPU_1499_20200922095734.pt'

     with st.spinner('Reconstructing the input image...'):

         if file is not None:
             if not os.path.exists(model_path):
                 gdd.download_file_from_google_drive(file_id='1AX0BRIYN5ty-nDwBWi7NE3XbB2QT0ucl', dest_path='./VAE_CPU_1499_20200922095734.pt', unzip=False)

         pil_img = Image.open(file).convert('RGB')
         img = TRANSFORMS(pil_img)
         img.unsqueeze_(0)

         VAE_Model = torch.jit.load('./VAE_CPU_1499_20200922095734.pt')
         with torch.no_grad():
             reconstructed_img, mu, logvar  = VAE_Model(img)
         display_img = reconstructed_img.squeeze(0)
         display_img = cv2.cvtColor(np.float32(display_img.permute(1, 2, 0)), cv2.COLOR_RGB2BGR)
         display_img = display_img[:, :, ::-1]
         st.image([pil_img, display_img],caption=['Input Img','Reconstructed Img'], width = 150)
