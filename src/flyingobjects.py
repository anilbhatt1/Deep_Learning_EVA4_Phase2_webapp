import streamlit as st
import torch
import os

from io import BytesIO
from PIL import Image

import src.home

from src.utils import mobilenet_classify, local_css
from google_drive_downloader import GoogleDriveDownloader as gdd

def write():
    """ Deep Learning Model to predict flying object """

    local_css("style.css")
    st.markdown("<h1 style='text-align: center; color: black;font-size: 40px;'>Flying Object Prediction</h1>", unsafe_allow_html=True)
    st.text('')
    st.text('')

    st.set_option('deprecation.showfileUploaderEncoding', False)
    #Mobilenet
    if st.checkbox("Predict flying objects"):
        st.subheader("Predicting flying objects using mobilenet")
        flying_object_classify()

#@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def flying_object_classify():

     class_names = ['Flying_Bird', 'Large_QuadCopter', 'Small_QuadCopter', 'Winged_Drone']

     file: BytesIO = st.file_uploader("Upload an image file", type=["jpg", "png"])
     if file is not None:
         gdd.download_file_from_google_drive(file_id='1-1Pv1sm0pXqwE_RZhHYuZSHaMunExS_G', dest_path='./flyingobject_model.pt', unzip=False)
         model = torch.jit.load('./flyingobject_model.pt')
         predicted = mobilenet_classify(model, file)
         class_idx = int((predicted[0][0]))
         st.markdown(f'#### Neural eyes have seen your image and believes it is a {class_names[class_idx]} !')

