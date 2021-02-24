"""Home page shown when the user enters the application"""
import streamlit as st
import os
from src.utils import local_css

def write():
    """Used to write the page in the main_app.py file"""
    with st.spinner("Loading Home ..."):
        local_css("style.css")
        st.markdown("<h1 style='text-align: center; color: black;font-size: 40px;'>Neural Eyes</h1>", unsafe_allow_html=True)
        st.text('')
        st.text('')
#        if "DYNO" in os.environ:
#            st.text('Running in Heroku')

        st.set_option('deprecation.showfileUploaderEncoding', False)
        st.write("""

                **Neural Eyes** is a deep-learning webapp focussed on computer-vision related applications. Deep-learning models used are trained in
                pytorch. Web-app is written using streamlit and hosted in heroku. You can find the **github** links in respective sections. Thank you
                for checking in !

                """
                )
        st.markdown("<h1 style='text-align: center; color: black;font-size: 20px;'> Thx for visiting...Please select an option for Neural Eyes to predict from side navigation bar</h1>", unsafe_allow_html=True)
