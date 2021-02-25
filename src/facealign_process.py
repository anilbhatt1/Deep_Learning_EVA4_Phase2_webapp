import streamlit as st
import torch
import os
import bz2
import urllib.request

from PIL import Image

import sys, cv2, dlib
import numpy as np
import src.faceblendcommon as fbc

#@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def align(file):

     model_file = './shape_predictor_68_face_landmarks.dat'

     with st.spinner('Aligning the faces...'):
      if file is not None:
         if not os.path.exists(model_file):
             urllib.request.urlretrieve("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2", f"{model_file}.bz2")

             # decompress data
             with bz2.open(f'{model_file}.bz2', 'rb') as f:
                 uncompressed_content = f.read()

             # write to file
             with open(model_file, 'wb') as f:
                 f.write(uncompressed_content)
                 f.close()

         pil_img = Image.open(file).convert('RGB')
         img = np.array(pil_img)
         img = img[:, :, ::-1].copy()   #Convert RGB to BGR for open-cv to work with

         # Create dlib facial landmark detector object
         detector = dlib.get_frontal_face_detector()
         predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

         # Detect landmarks
         points = fbc.getLandmarks(detector, predictor, img)
         points = np.array(points)

         # Set dimensions of o/p image
         h = 600
         w = 600

         # Normalize the image to O/P coordinates
         imNorm, points = fbc.normalizeImagesAndLandmarks((h, w), img, points)

         aligned_image = Image.fromarray(cv2.cvtColor(imNorm, cv2.COLOR_BGR2RGB))

         st.image([pil_img, aligned_image], width=200, caption=['Face', 'Aligned Face'])
