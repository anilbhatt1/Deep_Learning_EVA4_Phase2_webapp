import streamlit as st
import torch
import os

from io import BytesIO
from PIL import Image
from typing import Dict

from utils import mobilenet_classify, local_css, download_file_from_google_drive
from google_drive_downloader import GoogleDriveDownloader as gdd


def main():
    """ EVA4-Phase 2 Deep Learning Models """
    local_css("style.css")
    st.markdown("<h1 style='text-align: center; color: black;font-size: 40px;'>Neural Eyes</h1>", unsafe_allow_html=True)
    st.text('')
    st.text('')
    if "DYNO" in os.environ:
       st.text('Running Heroku')

    st.set_option('deprecation.showfileUploaderEncoding', False)
    #Mobilenet
    if st.checkbox("Predict flying objects"):
        st.subheader("Predicting flying objects using mobilenet")
        flying_object_classify()

def flying_object_classify():
     file: BytesIO = st.file_uploader("Upload an image file", type=["jpg", "png"])
     if file is not None:
         if "DYNO" in os.environ:
             st.text('Downloading model from drive..')
             gdd.download_file_from_google_drive(file_id='1-1Pv1sm0pXqwE_RZhHYuZSHaMunExS_G', dest_path='./flyingobject_model.pt', unzip=False)
             st.text('Download complete')
             model = torch.jit.load('./flyingobject_model.pt')
             st.text('Model loaded successfully')
         else:
             model = torch.load('flyingobject_model.pt')
             st.text('Model loaded successfully in local')
         predicted = mobilenet_classify(model, file)
         st.markdown(f'###Model identified uploaded image as {predicted}')

if __name__ == '__main__':
    main()
