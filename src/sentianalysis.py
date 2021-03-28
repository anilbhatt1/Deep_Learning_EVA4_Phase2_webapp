import streamlit as st

from io import BytesIO

from src.utils import local_css
import src.sentianalysis_process as sap

def write():
    """ Deep Learning NLP Model to analyse the sentiment from a given piece of text """

    local_css("style.css")
    st.markdown("<h1 style='text-align: center; color: black;font-size: 30px;'>Sentiment Analysis</h1>", unsafe_allow_html=True)
    st.write(
             """
             This app returns sentiment of an input review message. 3 possible outcomes are 'Positive', 'Neutral' Or 'Negative'.
             App uses an NLP model built based on CNNs. Model was trained on IMDB review data and hence will work best for
             text given in form of movie reviews. Tokenization is done using spacy and vocab of 25000 words built using glove.6B.100d.
             - [Github code reference(bentrevett)](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb)
             - [Github link for customized NLP model (used in this app)](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S9_Neural_Embeddings/E4P2S9_Convolutional_Sentiment_Analysis_cpu.ipynb)
             - [Github link for webapp](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2_webapp)
             """
             )

    st.set_option('deprecation.showfileUploaderEncoding', False)
    #Senti Analysis
    if st.checkbox("Start Sentiment Analysis"):
        senti_analysis()

#@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def senti_analysis():

     message = st.text_area('Enter your text', key='sentiment')
     if message is not None:
         
         st.text("""Sentiment is arrived as below:\nIf score <= 0.35         sentiment -> Negative\nif score > 0.35 & <= 0.6 sentiment -> Neutral\nif score > 0.6           sentiment -> Positive""")
         if st.button('GO'):
             sap.analyse(message)
