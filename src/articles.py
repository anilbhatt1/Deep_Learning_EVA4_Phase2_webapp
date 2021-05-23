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
        st.header('My Articles')
        st.write("""
                - [Understanding Object detection with YOLO](https://anilbhatt1.tech.blog/2020/07/03/understanding-object-detection-with-yolo/)

                - [CNN - Activation Functions, Global Average Pooling, Softmax, Negative Likelihood Loss](https://www.linkedin.com/pulse/cnn-activation-functions-global-average-pooling-softmax-n-bhatt/)

                - [Max-Pooling, Combining Channels using 1×1 convolutions, Receptive Field calculation](https://www.linkedin.com/pulse/max-pooling-combining-channels-using-11-convolutions-field-n-bhatt/)

                - [Convolutions - Work horse behind CNN](https://www.linkedin.com/pulse/convolutions-work-horse-behind-cnn-anilkumar-n-bhatt/)

                - [Understanding Receptive field in Computer Vision](https://www.linkedin.com/pulse/deep-learning-understanding-receptive-field-computer-n-bhatt/)

                - [How Computers classify objects in an image using Deep Learning](https://anilbhatt1.tech.blog/2020/01/30/how-computers-detect-objects-in-an-image-using-deep-learning/)

                """
                )
        st.header('Useful Articles (written by other authors)')
        st.write("""
                - [Differences between OpenCV, TF and PIL while reading and resizing images](https://towardsdatascience.com/image-read-and-resize-with-opencv-tensorflow-and-pil-3e0f29b992be)

                - [Understanding dimensions in PyTorch](https://towardsdatascience.com/understanding-dimensions-in-pytorch-6edf9972d3be)

                - [A Comprehensive Introduction to Different Types of Convolutions in Deep Learning](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215)

                 """
                )
