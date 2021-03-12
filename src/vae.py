import streamlit as st

from io import BytesIO

from src.utils import local_css
import src.vae_process as vp

def write():
    """ Deep Learning Model to reconstruct images of Indian car from input image supplied """

    local_css("style.css")
    st.markdown("<h1 style='text-align: center; color: black;font-size: 40px;'>Car Image Reconstruction using VAE</h1>", unsafe_allow_html=True)
    st.write(
             """
             Variational Auto-encoders(VAE) are special species of Auto-encoders(AE). AE typically will have an encoder and a decoder network.
             Encoder network will create latent vector/bottleneck from given input image.Decoder network will take the bottleneck and recontruct the image.
             Usecases of auto-encoders includes denoising the image, reconstruction of poor resolution images etc. 
             However, auto-encoders cant seamlessly interpolate between classes. This is where VAEs come into picture.
             VAE's latent spaces are, by design, continuous, allowing easy random sampling and interpolation.
             Instead of predicting a point as what vanilla autocoders do, VAE predicts a cloud of points. 
             Here, VAEs are employed to reconstruct image of an input Indian car supplied.
             - [Github link for model](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/tree/master/S7_VAE)
             - [Github link for webapp](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2_webapp)
             """
             )

    st.set_option('deprecation.showfileUploaderEncoding', False)
    #VAE
    if st.checkbox("Upload image to reconstruct"):
        vae_image()

#@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def vae_image():

     file_key  = 'vae'

     file: BytesIO = st.file_uploader("jpg or png formats only, Upload Left front facing car images with white or no background", type=["jpg", "png"], key=file_key)
     if file is not None:
         st.text('')
         if st.checkbox("Reconstruct the car"):
             vp.generate(file)
