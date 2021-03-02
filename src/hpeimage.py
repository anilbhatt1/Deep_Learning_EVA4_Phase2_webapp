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
             Face Align app will align the face given to it as input. App uses opencv and dlib-68-face-landmark detector.
             Practical applications includes aligning the faces to make it front-facing for crime investigations and realistic 
             image synthesis that can be used for AI model training for computer vision applications.
             - [Github link for model](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/tree/master/S3_Facial%20Landmark%20Detection_Alignment_Swap)
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

     file: BytesIO = st.file_uploader("Upload the human image", type=["jpg", "png"], key=file_key)
     if file is not None:
         st.text('')
         if st.checkbox("Estimate the pose"):
             hpi.estimate(file)
