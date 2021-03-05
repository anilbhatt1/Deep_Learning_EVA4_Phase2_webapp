"""Article page shown when the user enters the application"""
import streamlit as st
from src.utils import local_css

def write():
    """Used to save the article links"""
    with st.spinner("Loading Articles ..."):
        local_css("style.css")
        st.markdown("<h1 style='text-align: center; color: black;font-size: 40px;'>Articles</h1>", unsafe_allow_html=True)
        st.text('')
        st.text('')
        st.set_option('deprecation.showfileUploaderEncoding', False)
        st.write("""

                Will update the links to articles that I written on DNN related topics soon !

                """
                )
