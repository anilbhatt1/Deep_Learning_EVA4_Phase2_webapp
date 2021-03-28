import streamlit as st

from io import BytesIO

from src.utils import local_css
import src.multiqnanalysis_process as map

def write():
    """ Deep Learning NLP Model to predict the type of question given as input text """

    local_css("style.css")
    st.markdown("<h1 style='text-align: center; color: black;font-size: 30px;'>Multi-Question Classification</h1>", unsafe_allow_html=True)
    st.write(
             """
             This app will classify which category a particular input question belongs to. 6 types of qn are there : Questions about Humans, Entities, 
             Description, Location, Numercial & Abbreviation. 
             App uses an NLP model built based on CNNs. Model was trained on TREC dataset with 6 labels mentioned above. 
             Tokenization is done using spacy and vocab of 25000 words built using glove.6B.100d.
             - [Github code reference(bentrevett)](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb)
             - [Github link for customized NLP model (used in this app)](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/S9_Neural_Embeddings/E4P2S9_Convolutional_Sentiment_Analysis_cpu.ipynb)
             - [Github link for webapp](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2_webapp)
             """
             )

    st.set_option('deprecation.showfileUploaderEncoding', False)
    #Multi-Qn Analysis
    if st.checkbox("Classify the question"):
        multiqn_analysis()

#@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def multiqn_analysis():

     message = st.text_area('Enter your text', key='question')
     if message is not None:

         st.text("""Question will be classified as one of the following : Entities, Humans, Description, \nNumerical, Location or Abbreviation""")
         if st.button('GO'):
             map.analyse(message)
