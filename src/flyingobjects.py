import streamlit as st
import torch
import os
import os.path as osp
from pathlib import Path

from io import BytesIO
from PIL import Image

import src.home

from src.utils import mobilenet_classify, local_css
from google_drive_downloader import GoogleDriveDownloader as gdd

def write():
    """ Deep Learning Model to predict flying object """

    local_css("style.css")
    st.markdown("<h1 style='text-align: center; color: black;font-size: 40px;'>Flying Object Prediction</h1>", unsafe_allow_html=True)
    st.write(
             """
             Deep-learning app to help predict the flying objects in our skies. **Mobilenet** model used here is trained against 4 objects
             : **flying birds, large quadcopters, small quadcopters and winged drones**. App can be enhanced by training the model including 
             more flying objects. Practical application includes distinguishing between innocous flying objects and inimical ones based
             on context. For example, a flying bird over football stadium may be an innocous one whereas same flying bird will be inimical
             over an airport runway.
             - [Github link for model](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/tree/master/S2_Mobilenet_QuadCopters_Lambda)
             - [Github link for webapp](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2_webapp)
             """
             )
    st.text('')

    st.set_option('deprecation.showfileUploaderEncoding', False)
    #Mobilenet
    if st.checkbox("Predict flying objects"):
        st.subheader("Predicting flying objects using mobilenet")
        flying_object_classify()

#@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def flying_object_classify():

     class_names = ['Flying_Bird', 'Large_QuadCopter', 'Small_QuadCopter', 'Winged_Drone']
     file_key  = 'flying_object'

     file: BytesIO = st.file_uploader("Upload an image file", type=["jpg", "png"], key=file_key)
     if file is not None:
         model_path = Path("./flyingobject_model.pt")
         if not model_path.exists():
             gdd.download_file_from_google_drive(file_id='1-1Pv1sm0pXqwE_RZhHYuZSHaMunExS_G', dest_path='./flyingobject_model.pt', unzip=False)
         model = torch.jit.load('./flyingobject_model.pt')
         predicted = mobilenet_classify(model, file)
         class_idx = int((predicted[0][0]))
         st.markdown(f'### Neural eyes have seen your image and believes it is a {class_names[class_idx]} !')
         st.text('')
         st.image(Image.open(file), use_column_width=True)
