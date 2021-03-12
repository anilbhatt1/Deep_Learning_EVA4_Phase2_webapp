import streamlit as st
import torch
import torchvision
from torchvision import transforms
import os
import bz2
import urllib.request
import onnxruntime
import re
from operator import itemgetter

from PIL import Image

import sys, cv2
import numpy as np
from google_drive_downloader import GoogleDriveDownloader as gdd


mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
toTensor = transforms.Compose([transforms.ToTensor(), 
                               transforms.Normalize(mean, std)])


threshold      = 0.6
joints = ['0 - r ankle',     '1 - r knee',      '2 - r hip', 
          '3 - l hip',       '4 - l knee',      '5 - l ankle', 
          '6 - pelvis',      '7 - thorax',      '8 - upper neck', 
          '9 - head top',    '10 - r wrist',    '11 - r elbow', 
          '12 - r shoulder', '13 - l shoulder', '14 - l elbow', 
          '15 - l wrist']
joints = [re.sub(r'[0-9]+|-', '', joint).strip().replace(' ', '-') for joint in joints]

pose_pairs = [
# UPPER BODY
              [9, 8],
              [8, 7],
              [7, 6],
# LOWER BODY
              [6, 2],
              [2, 1],
              [1, 0],

              [6, 3],
              [3, 4],
              [4, 5],
# ARMS
              [7, 12],
              [12, 11],
              [11, 10],

              [7, 13],
              [13, 14],
              [14, 15]
              ]

get_keypoints = lambda pose_layers: map(itemgetter(1, 3), [cv2.minMaxLoc(pose_lyr) for pose_lyr in pose_layers])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def onnx_get_image_with_points_connected_lines(image_p, out_ht, out_wd, ort_outs):
    pose_layers = ort_outs[0][0]
    key_points = list(get_keypoints(pose_layers=pose_layers))
    is_joint_plotted = [False for i in range(len(joints))]
    for pose_pair in pose_pairs:
        from_j, to_j = pose_pair

        from_thr, (from_x_j, from_y_j) = key_points[from_j]
        to_thr, (to_x_j, to_y_j)       = key_points[to_j]

        img_height, img_width, _       = image_p.shape

        from_x_j, to_x_j               = from_x_j * (img_width / out_wd), to_x_j * (img_width / out_wd)
        from_y_j, to_y_j               = from_y_j * (img_height / out_ht), to_y_j * (img_height / out_ht)
        from_x_j, to_x_j               = int(from_x_j), int(to_x_j)
        from_y_j, to_y_j               = int(from_y_j), int(to_y_j)

        if from_thr > threshold and not is_joint_plotted[from_j]:
            # this is a joint
            cv2.ellipse(image_p, (from_x_j, from_y_j), (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            is_joint_plotted[from_j] = True

        if to_thr > threshold and not is_joint_plotted[to_j]:
            # this is a joint
            cv2.ellipse(image_p, (to_x_j, to_y_j), (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            is_joint_plotted[to_j] = True

        if from_thr > threshold and to_thr > threshold:
           # this is a joint connection, plot a line
           cv2.line(image_p, (from_x_j, from_y_j), (to_x_j, to_y_j), (255, 74, 0), 2)

    return image_p

def estimate(file):
     model_path = './quantized_model_1.onnx'

     with st.spinner('Estimating the pose...'):

      if file is not None:
         if not os.path.exists(model_path):
             gdd.download_file_from_google_drive(file_id='112b29KLW1oNgSHDDlHEJzZXVGy62B76a', dest_path='./quantized_model_1.onnx', unzip=False)
             st.text('Model downloaded')

         pil_img = Image.open(file).convert('RGB')
         img = np.array(pil_img)
         img = img[:, :, ::-1].copy()   #Convert RGB to BGR for open-cv to work with
         img = cv2.resize(img, (256, 256))
         img_resize = toTensor(img).unsqueeze(0) # This will add a dimension of 1 to fake a batch

         # Creating onnxruntime session
         ort_session = onnxruntime.InferenceSession(model_path)

         # Setting ort_inputs to be fed to onnx quantized model
         ort_inputs = {ort_session.get_inputs()[0].name: np.squeeze(to_numpy(img_resize.unsqueeze(0)), axis=0)}  #unsqueeze to remove the dimension we faked

         # Output prediction by calling ort_session
         ort_outs = ort_session.run(None, ort_inputs)

         out_ht, out_wd = ort_outs[0][0].shape[1], ort_outs[0][0].shape[2]

         connected_img = onnx_get_image_with_points_connected_lines(img, out_ht, out_wd, ort_outs)

         pose_img = connected_img[:, :, ::-1]  # Converting to RGB

         st.image([pil_img, pose_img], caption=['Input Img', 'Estimated Pose'])
