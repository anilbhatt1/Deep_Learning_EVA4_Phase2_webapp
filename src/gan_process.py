import streamlit as st
import torch
import torchvision
from torchvision import transforms
import os

from PIL import Image

import sys, cv2
import numpy as np
from google_drive_downloader import GoogleDriveDownloader as gdd
import random
import pandas as pd 

n_noise   = 100

# Downloading generator model & loading generator with this model
G_path = './G_CPU_999_20200910104719.pt'
if not os.path.exists(G_path):
    gdd.download_file_from_google_drive(file_id='1YgB6vBw4iS1n4LgOau-BrX0UhLVC9PZw', dest_path='./G_CPU_999_20200910104719.pt', unzip=False)
G = torch.jit.load('./G_CPU_999_20200910104719.pt')

'''
This will generate a tensor of  shape (32, 100). Values will be as follows:
Flag = 'positive' , tensor values will be between 0 and 1
Flag = 'negative' , tensor values will be between -1 and 0 
Flag = 'normal', tensor values will be between 1 and -1 (Default option)
'''
def tensor_gen(size, n_noise, range_values):

    if size > 1:
        rand_tensor = torch.randn(size, n_noise)                        # This will generate (32, 100) tensor ie 32 sets of 100 random normal numbers
    else:
        min, max = range_values                                         # This will generate (1, 100) tensor ie 1 set of 100 random normal numbers
        rand_tensor = (max-min)*torch.rand((size, n_noise)) + min

    return rand_tensor

def get_sample_image(G, size, z):

    y_hat = G(z).view(size, 3, 128, 128).permute(0, 2, 3, 1)    # Generate the image based on tensor size & modify axes to (32, 128, 128, 3) via permute
    result = (y_hat.detach().cpu().numpy()+1)/2.
    return result

def display_values(range_values, z):
    st.markdown("### Random tensor values for the range you selected are as below. Please click on 'GO' button below to generate the car image")
    df = z.numpy().reshape(10,10)
    df = pd.DataFrame(df).astype("float")
    st.table(df)

def generate():

     with st.spinner('Generating the car images...'):

         size    = 32
         range_values    = ()  # Passing an empty tuple as car generation for set of images is not based on range of values supplied as input
         z      = tensor_gen(size, n_noise, range_values)
         result = get_sample_image(G, size, z)
         # Generate 5 random integers. These integers will used as indexes to select and display the car images from 'result'
         lst = []
         for i in range(5):
             num = random.randint(0, 31)
             lst.append(num)

         img0 = cv2.cvtColor(result[lst[0]], cv2.COLOR_BGR2RGB)  # Converting to RGB format
         img1 = cv2.cvtColor(result[lst[1]], cv2.COLOR_BGR2RGB)
         img2 = cv2.cvtColor(result[lst[2]], cv2.COLOR_BGR2RGB)
         img3 = cv2.cvtColor(result[lst[3]], cv2.COLOR_BGR2RGB)
         img4 = cv2.cvtColor(result[lst[4]], cv2.COLOR_BGR2RGB)
         #img0 = result[lst[0]][:, :, ::-1]   # Converting to RGB format
         st.image([img0, img1, img2, img3, img4])
         st.markdown('### Generated Car Images')

def explore():

    # Displaying the grid values based on values selected
    range_values = st.slider("Select the range of values for which you want to generate car image", -1.0 , 1.0, (-0.5, 0.5))
    size = 1
    z    = tensor_gen(size, n_noise, range_values)
    display_values(range_values, z)

    if st.button('GO'):
        result = get_sample_image(G, size, z)
        img0 = result[0][:, :, ::-1]
        st.image(img0, caption = 'Generated Car Image')
