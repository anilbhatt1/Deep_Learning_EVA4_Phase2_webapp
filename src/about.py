"""Home page shown when the user enters the application"""
import streamlit as st
from src.utils import local_css
from PIL import Image

def write():
    """Used to write about page"""
    with st.spinner("Loading Home ..."):
        local_css("style.css")
        st.markdown("<h1 style='text-align: left; color: black;font-size: 40px;'> Who Am I ? </h1>", unsafe_allow_html=True)
        st.text('')
        st.text('')

        img = Image.open('./About_pic.jpg')
        st.image(img, width = 200)
        st.set_option('deprecation.showfileUploaderEncoding', False)
        st.write("""

                I am Anil Bhatt from India. A deep learning enthusiast who loves to try out practical applications of AI. 
                Loves reading, trekking and watching football. Coordinates 
                to reach me are as listed below. Thx for checking in. Have a great day !
                - [Github](https://github.com/anilbhatt1)
                - [Linkedin](https://www.linkedin.com/in/anilkumar-n-bhatt/)
                """
                )
        i = 0
        for i in range(5):
            st.text(' ')
        st.markdown("<h1 style='text-align: center; color: white;font-size: 16px;'> ' Fool didn't know it was impossible, so he did it ! ' - Unknown </h1>", unsafe_allow_html=True)
