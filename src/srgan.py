import streamlit as st

from io import BytesIO

from src.utils import local_css
import src.srgan_process as srp

def write():
    """ Deep Learning Model to create better resolution images from low resolution input images supplied """

    local_css("style.css")
    st.markdown("<h1 style='text-align: center; color: black;font-size: 30px;'>Better Resolution images using SRGAN</h1>", unsafe_allow_html=True)
    st.write(
             """
             Image Super-Resolution (SR) refers to the process of recovering high-resolution(HR) images from low-resolution(LR) images.
             One promising approach to generate HR images is using DNNs with 
             appropriate loss functions. In this app, we are focusing on one such method termed as SRGAN (Super Resolution Generative Adverserial Network).
             Like GANs, SRGANs too use generator and discriminator networks, however loss functions vary. SRGANs are developed based on the paper
             'Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network'. As the customized SRGAN model used here is trained 
             majorly on 4 objects - **birds, small quadcopters, large quadcopters and winged drones** - supplying images corresponding to these will 
             yield better results. Real-life usecases includes medical imaging, surveillance etc. Github link references as listed below.
             - [Arxiv link for 'Photo-Realistic Single Image Super-Resolution Using a GAN' paper](https://arxiv.org/abs/1609.04802)
             - [Github code reference for SRGAN](https://github.com/leftthomas/SRGAN)
             - [Github link for customized SRGAN model (used in this app)](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/tree/master/S8_SRGAN_Neural%20Transfer)
             - [Github link for webapp](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2_webapp)
             """
             )

    st.set_option('deprecation.showfileUploaderEncoding', False)
    #SRGAN
    if st.checkbox("Upload image"):
        srgan_image()

#@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def srgan_image():

     file_key  = 'srgan'

     file: BytesIO = st.file_uploader("jpg or png format only", type=["jpg", "png"], key=file_key)
     if file is not None:
         st.text('')
         if st.checkbox("Generate better resolution image"):
             srp.generate(file)
