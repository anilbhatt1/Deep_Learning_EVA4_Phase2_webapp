import streamlit as st
import torch
import torchvision
from torchvision import transforms
import os
import bz2
import urllib.request
import onnxruntime
import re
import ffmpeg
from operator import itemgetter

from PIL import Image

import sys, cv2
import numpy as np
from google_drive_downloader import GoogleDriveDownloader as gdd
import tempfile
import datetime

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
toTensor = transforms.Compose([transforms.ToTensor(), 
                               transforms.Normalize(mean, std)])


threshold      = 0.3
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

def onnx_get_frame_with_points_connected_lines(image_p, out_ht, out_wd, ort_outs):
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

     # Removing output files created from previous execution (if they still exists)
     if os.path.exists("./output_pose.avi"):
         os.remove("./output_pose.avi")

     if os.path.exists("./output_pose.mp4"):
         os.remove("./output_pose.mp4")

     with st.spinner('Estimating the pose and preparing video...'):

      if file is not None:
         if not os.path.exists(model_path):
             gdd.download_file_from_google_drive(file_id='112b29KLW1oNgSHDDlHEJzZXVGy62B76a', dest_path='./quantized_model_1.onnx', unzip=False)
             st.text('Model downloaded')

         # Creating onnxruntime session
         ort_session = onnxruntime.InferenceSession(model_path)

         # Reading input file to a tempfile for opencv to work with
         tfile = tempfile.NamedTemporaryFile(delete=True) # Setting delete = True to ensure that tfile is deleted upon its closure
         tfile.write(file.read())
         cap = cv2.VideoCapture(tfile.name)
         # Capturing frame_size to create videowriter object
         while(cap.isOpened()):
             ret, frame = cap.read()
             frame_size = (frame.shape[1], frame.shape[0])
             break

         time_now = datetime.datetime.now()
         st.markdown(f'Starting process of capture {time_now}')

         # Define the codec and create VideoWriter object
         fourcc = cv2.VideoWriter_fourcc(*'XVID')
         out = cv2.VideoWriter('./output_pose.avi',fourcc, 20.0, frame_size)

         # Reads video frame-by-frame, predicts pose for each frame and writes back to output video
         time_now = datetime.datetime.now()
         st.markdown(f'Started creating frames {time_now}')

         while(cap.isOpened()):
             ret, frame = cap.read()
             if ret == True:
                 img = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                 img_resize = cv2.resize(img, (256, 256))                 # Resizing the input image frame as we are using Resnet 256x256
                 x = toTensor(img_resize).unsqueeze(0)
                 ort_inputs = {ort_session.get_inputs()[0].name: np.squeeze(to_numpy(x.unsqueeze(0)), axis=0)} # Preparing input for onnx model to predict
                 ort_outs = ort_session.run(None, ort_inputs)
                 out_ht, out_wd = ort_outs[0][0].shape[1], ort_outs[0][0].shape[2]
                 image_p = onnx_get_frame_with_points_connected_lines(img, out_ht, out_wd, ort_outs)
                 image_p = cv2.cvtColor(image_p, cv2.COLOR_RGB2BGR)
                 out.write(image_p)
             else:
                 break

         time_now = datetime.datetime.now()
         st.markdown(f'Finished creating frames {time_now}')

         cap.release()
         out.release()
         tfile.close()

         time_now = datetime.datetime.now()
         st.markdown(f'Started converting to mp4 {time_now}')

         os.system("ffmpeg -y -i ./output_pose.avi -vcodec libx264 ./output_pose.mp4")  #Converting from avi to mp4 for display as st.video() doesn't work on avi

         time_now = datetime.datetime.now()
         st.markdown(f'Finished converting to mp4 {time_now}')

         # Displaying Video output
         vid_file = open("./output_pose.mp4", 'rb').read()
         st.video(vid_file)
