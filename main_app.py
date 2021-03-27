import streamlit as st

import src
import src.home
import src.flyingobjects
import src.faceswap
import src.facealign
import src.hpeimage
import src.hpevideo
import src.gan
import src.vae
import src.srgan
import src.styletransfer
import src.sentianalysis
import src.articles
import src.about

from src.utils import local_css

PAGES = {
        "Home":src.home,
        "Flying Objects Prediction":src.flyingobjects,
        "Face Swap":src.faceswap,
        "Face Align":src.facealign,
        "Human Pose Estimation(Image)":src.hpeimage,
        "Human Pose Estimation(Video)":src.hpevideo,
        "GAN - Car Image Generation":src.gan,
        "VAE - Car Image Reconstruction":src.vae,
        "SRGAN - Image Super Resolution":src.srgan,
        "Style Transfer":src.styletransfer,
        "Sentiment Analysis":src.sentianalysis,
        "Articles":src.articles,
        "About":src.about
        }

def main():
    """ Main function of the Neural Eyes EVA4-Phase 2 Deep Learning Webapp """

    #st.sidebar.header('Navigation')
    selection = st.sidebar.radio('Go To', list(PAGES.keys()))

    page = PAGES[selection]
    with st.spinner(f'Loading {selection}...'):
        write_page(page)

def write_page(page):
    """Writes the specified page/module
       Our multipage app is structured into sub-files with a `def write()` function
       Arguments:
        page {module} -- A module with a 'def write():' function
    """
    page.write()

if __name__ == '__main__':
    main()
