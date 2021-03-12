import streamlit as st

from io import BytesIO

from src.utils import local_css
import src.gan_process as gp

def write():
    """ Deep Learning Model to generate images of Indian car from random vector supplied """

    local_css("style.css")
    st.markdown("<h1 style='text-align: center; color: black;font-size: 40px;'>Car Image Generation using GAN</h1>", unsafe_allow_html=True)
    st.write(
             """
             Generative Adverserial Networks (GANs) are neural networks that are trained in an adversarial manner to generate data by mimicking some
             distribution. GANs are comprised of two neural networks, pitted one against the other (hence the name adversarial). Applications of GANs are vast
             that include Generate Examples for Image Datasets, Generate Photographs of Human Faces, Generate Realistic Photographs, Generate Cartoon Characters, 
             Drug Research etc. Here, GANs are employed to generate images of Indian cars based on input vector values. DCGAN is the type of GAN used.
             - [Reference link for approach followed](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/tree/master/S6_GAN/Model%20Weights/Yangyangii%20Cars%20Example%20Github)
             - [Github link for model](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/tree/master/S6_GAN)
             - [Github link for webapp](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2_webapp)
             """
             )

    st.set_option('deprecation.showfileUploaderEncoding', False)
    #GAN
    if st.checkbox("Car Image generation for given range of values"):
        gp.explore()
    elif st.checkbox("Random Car Image generation"):
        gp.generate()
