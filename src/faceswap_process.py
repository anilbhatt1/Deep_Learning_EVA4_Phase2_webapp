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
def swap(file1, file2):

     model_file = './shape_predictor_68_face_landmarks.dat'

     with st.spinner('Swapping the faces...'):
      if file1 is not None and file2 is not None:
         if not os.path.exists(model_file):
             urllib.request.urlretrieve("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2", f"{model_file}.bz2")

             # decompress data
             with bz2.open(f'{model_file}.bz2', 'rb') as f:
                 uncompressed_content = f.read()

             # write to file
             with open(model_file, 'wb') as f:
                 f.write(uncompressed_content)
                 f.close()

         pil_img1 = Image.open(file1).convert('RGB')
         img1 = np.array(pil_img1)
         img1 = img1[:, :, ::-1].copy()   #Convert RGB to BGR for open-cv to work with

         pil_img2 = Image.open(file2).convert('RGB')
         img2 = np.array(pil_img2)
         img2 = img2[:, :, ::-1].copy()   #Convert RGB to BGR for open-cv to work with

         img1Warped = np.copy(img2)

         # Create dlib facial landmark detector object
         detector = dlib.get_frontal_face_detector()
         predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

         # Read array of corresponding points
         points1 = fbc.getLandmarks(detector, predictor, img1)
         points2 = fbc.getLandmarks(detector, predictor, img2)

         # Find convex hull
         hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)

         # Create convex hull lists
         hull1 = []
         hull2 = []
         for i in range(0, len(hullIndex)):
             hull1.append(points1[hullIndex[i][0]])
             hull2.append(points2[hullIndex[i][0]])

         # Calculate Mask for Seamless cloning
         hull8U = []
         for i in range(0, len(hull2)):
             hull8U.append((hull2[i][0], hull2[i][1]))

         mask = np.zeros(img2.shape, dtype=img2.dtype) 
         cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

         # Find Centroid
         m = cv2.moments(mask[:,:,1])
         center = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))

         # Find Delaunay traingulation for convex hull points
         sizeImg2 = img2.shape
         rect = (0, 0, sizeImg2[1], sizeImg2[0])

         dt = fbc.calculateDelaunayTriangles(rect, hull2)

         # If no Delaunay Triangles were found, quit
         if len(dt) == 0:
             quit()

         tris1 = []
         tris2 = []
         for i in range(0, len(dt)):
             tri1 = []
             tri2 = []
             for j in range(0, 3):
                 tri1.append(hull1[dt[i][j]])
                 tri2.append(hull2[dt[i][j]])

             tris1.append(tri1)
             tris2.append(tri2)

         # Simple Alpha Blending
         # Apply affine transformation to Delaunay triangles
         for i in range(0, len(tris1)):
             fbc.warpTriangle(img1, img1Warped, tris1[i], tris2[i])

         # Clone seamlessly.
         output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)
         swapped_face = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display in streamlit UI

         st.image([pil_img1, pil_img2,swapped_face], width=200, caption=['Face1', 'Face2', 'Swapped Face'])
