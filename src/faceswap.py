import streamlit as st

from io import BytesIO

from src.utils import local_css
import src.faceswap_process as fs

def write():
    """ Deep Learning Model to swap the faces """

    local_css("style.css")
    st.markdown("<h1 style='text-align: center; color: black;font-size: 40px;'>Face Swap</h1>", unsafe_allow_html=True)
    st.write(
             """
             Face swap app will merge 2 faces given to it. App uses opencv, dlib-68-face-landmark detector and works based on Delaunay triangulation.
             Practical applications include genertaing new faces that can be used for AI model training & testing and
             realistic image synthesis that can be shown to patients who plan to undergo cosmetic surgeries.
             - [Github link for model](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/tree/master/S3_Facial%20Landmark%20Detection_Alignment_Swap)
             - [Github link for webapp](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2_webapp)
             """
             )

    st.set_option('deprecation.showfileUploaderEncoding', False)
    #Mobilenet
    if st.checkbox("Upload faces to swap"):
        st.subheader("Upload the images")
        face_swap()

#@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def face_swap():

     file1_key  = 'face1'
     file2_key  = 'face2'

     file1: BytesIO = st.file_uploader("Upload first face, Use camera facing images for best results", type=["jpg", "png"], key=file1_key)
     file2: BytesIO = st.file_uploader("Upload second face,Use camera facing images for best results", type=["jpg", "png"], key=file2_key)
     if file1 is not None and file2 is not None:
         st.text('')
         if st.checkbox("Swap the faces"):
             fs.swap(file1, file2)
