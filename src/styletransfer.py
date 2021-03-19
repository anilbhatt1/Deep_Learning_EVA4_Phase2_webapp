import streamlit as st

from io import BytesIO

from src.utils import local_css
import src.styletransfer_process as stp

def write():
    """ Deep Learning Model to create style transfer images """

    local_css("style.css")
    st.markdown("<h1 style='text-align: center; color: black;font-size: 40px;'>Style Transfer using Fast-Neural Style</h1>", unsafe_allow_html=True)
    st.write(
             """
             Neural-Style or Neural-Transfer, allows us to take an image and reproduce it with a new artistic style. In this app, we are using Fast-Neural 
             style with the help of pre-trained models that belong to 4 styles - Mosaic, Udnie, Candy & Rain-princess. Input image supplied can be 
             converted to any of these 4 styles. Real-life usecase of style transfer includes converting night-vision images to day light settings or viceversa.
             - [Reference - Fast Neural Style](https://github.com/pytorch/examples/tree/6c8e2bab4d45f2386929c83bb4480c18d2b660fd/fast_neural_style)
             - [Github link for model](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/tree/master/S8_SRGAN_Neural%20Transfer)
             - [Github link for webapp](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2_webapp)
             """
             )

    st.set_option('deprecation.showfileUploaderEncoding', False)
    #STYLE TRANSFER
    if st.checkbox("Upload image"):
        style_image()

#@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def style_image():

     file_key  = 'style'

     file: BytesIO = st.file_uploader("jpg or png formats only", type=["jpg", "png"], key=file_key)
     if file is not None:
         st.text('')
         if st.checkbox('Start Style Transfer'):
            stp.generate(file)
