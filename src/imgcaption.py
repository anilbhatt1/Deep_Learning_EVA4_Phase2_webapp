import streamlit as st

from io import BytesIO

from src.utils import local_css
import src.imgcaption_process as icp

def write():
    """ Deep Learning Model to caption input images supplied """

    local_css("style.css")
    st.markdown("<h1 style='text-align: center; color: black;font-size: 30px;'>Image Captioning</h1>", unsafe_allow_html=True)
    st.write(
             """
             Image Captioning refers to the generation of a descriptive caption for an input image supplied.
             Model implemented here is based on "Show, Attend and Tell" paper. Idea is to give an image to the model and model predicts a caption based on
             image. Encoder-decoder architecture with attention is used. Encoder used to encode the input image is Resnet-18. These encodings
             are fed to the decoder. Job of decoder is to generate the caption based on the encodings. Decoder is an LSTM based network that uses 
             attention mechanism. Attention mechanism helps the LSTM to focus on 
             specific parts of input image (encodings) based on the weight provided for each pixel. Model is trained on flickr8k dataset.  
             - [Arxiv link for 'Show, Attend and Tell' Paper](https://arxiv.org/abs/1502.03044)
             - [Original Github code reference](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
             - [Github link for customized model (used in this app)](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/tree/master/S12_Image_Captioning_and_text_to_images)
             - [Github link for webapp](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2_webapp)
             """
             )

    st.set_option('deprecation.showfileUploaderEncoding', False)
    #SRGAN
    if st.checkbox("Upload image"):
        imgcaption_image()

#@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def imgcaption_image():

     file_key  = 'imgcap'

     file: BytesIO = st.file_uploader("jpg or png format only", type=["jpg", "png"], key=file_key)
     if file is not None:
         st.text('')
         if st.checkbox("Caption the input image"):
             icp.generate(file)
