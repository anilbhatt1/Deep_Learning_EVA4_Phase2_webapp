import streamlit as st

from io import BytesIO

from src.utils import local_css
import src.hpeimage_process as hpi

def write():
    """ Deep Learning Model to estimate the pose of input human image """

    local_css("style.css")
    st.markdown("<h1 style='text-align: center; color: black;font-size: 40px;'>Human Pose Estimation (Images)</h1>", unsafe_allow_html=True)
    st.write(
             """
             Human pose estimation (HPE) is the process of estimating the configuration of the body (pose) from a single, typically monocular, image.
             It can be applied to many applications such as action/activity recognition, action detection, human tracking, in movies and animation, 
             virtual reality, human-computer interaction, video surveillance, medical assistance, self-driving, sports motion analysis, etc.
             HPE Image app will give back the pose of input image with human joints connected. App uses quantized ONNX ResNet model trained based on
             'Simple Baseline for HPE and tracking' paper.
             - [Arxiv link for Simple Baseline for HPE and Tracking paper](https://arxiv.org/pdf/1804.06208.pdf)
             - [Github link for model](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/tree/master/S5_Human_Pose_Estimation)
             - [Github link for webapp](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2_webapp)
             """
             )

    st.set_option('deprecation.showfileUploaderEncoding', False)
    #Mobilenet
    if st.checkbox("Upload image to estimate pose"):
        st.subheader("Upload the image")
        hpe_image()

#@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def hpe_image():

     file_key  = 'human'

     file: BytesIO = st.file_uploader("jpg or png formats only", type=["jpg", "png"], key=file_key)
     if file is not None:
         st.text('')
         if st.checkbox("Estimate the pose"):
             print('Estimating pose')
             hpi.estimate(file)
