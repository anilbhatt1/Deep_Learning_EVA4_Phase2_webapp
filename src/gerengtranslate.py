import streamlit as st

from io import BytesIO

from src.utils import local_css
import src.gerengtranslate_process as tran

def write():
    """ Deep Learning NLP Model to translate the german text to english """

    local_css("style.css")
    st.markdown("<h1 style='text-align: center; color: black;font-size: 30px;'>German English Translator</h1>", unsafe_allow_html=True)
    st.write(
             """
             This app translates the given input german text to english. This is based on “Neural Machine Translation by Jointly Learning to Align and Translate” of Bahdanau et al. (2015)
             App uses an encoder-decoder with attention architecture. Tokenization is done using spacy and vocab built using IWSLT dataset.
             - [Github code reference(bentrevett)](https://bastings.github.io/annotated_encoder_decoder/)
             """
             )

    st.set_option('deprecation.showfileUploaderEncoding', False)
    #Senti Analysis
    if st.checkbox("Start Translation"):
        translate_text()

#@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def translate_text():

     message = st.text_area('Enter your german text', key='german')
     if message is not None:

         st.text("""Currently app can translate only german text to english and not vice-versa""")
         if st.button('GO'):
             tran.translate(message)
