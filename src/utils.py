from io import BytesIO
from typing import List, Tuple, Dict

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import streamlit as st

from PIL import Image
from torchvision.transforms import Compose

Mean: List = [0.48043839, 0.44820218, 0.39760034]
STD: List = [0.27698959, 0.26908774, 0.28216029]
CLASS_NAMES = ['Flying Birds', 'Large QuadCopters', 'Small QuadCopters', 'Winged Drones']

TRANSFORMS: Compose = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=Mean,std=STD),
    ])

def mobilenet_classify(model: nn.Module, image: BytesIO):
    img: Image  = Image.open(image).convert('RGB')
    img: Tensor = TRANSFORMS(img)
    img.unsqueeze_(0)

    model.eval()
    with torch.no_grad():
        labels_pred: Tensor = model(img)
        labels_pred_max  = labels_pred.argmax(dim =1, keepdim = True)

    labels_pred_max = labels_pred_max.cpu().numpy()
    return labels_pred_max

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

