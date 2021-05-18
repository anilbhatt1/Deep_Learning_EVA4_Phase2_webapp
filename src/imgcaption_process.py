import streamlit as st
import torch
import torchvision
from torchvision import transforms as T
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from PIL import Image

import sys, cv2
from google_drive_downloader import GoogleDriveDownloader as gdd

def generate(file):
     encoder_path = './Test_flickr8k_caption.encoder.scripted.pt'
     decoder_path = './Test_flickr8k_caption.decoder.scripted.pt' 
     wordmap_path = './WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json'

     with st.spinner('Captioning the input image...'):

         if file is not None:
             if not os.path.exists(encoder_path):
                 gdd.download_file_from_google_drive(file_id='1I-PAvJB3-qVjF2WEOBWnNX48KmxIQ6l0', dest_path=encoder_path, unzip=False)
             if not os.path.exists(decoder_path):
                 gdd.download_file_from_google_drive(file_id='1EnNif08a9SmhTWKSEY0GRPAZW6v-IJHN', dest_path=decoder_path, unzip=False)
             if not os.path.exists(wordmap_path):
                 gdd.download_file_from_google_drive(file_id='1-2MmHrIBCVpzuPZWQKPk_X8ebG23Yfke', dest_path=wordmap_path, unzip=False)

         pil_img = Image.open(file).convert('RGB')

         encoder = torch.jit.load(encoder_path)
         decoder = torch.jit.load(decoder_path)

         with open(wordmap_path, "r") as j:
             word_map = json.load(j)
             rev_word_map = {v: k for k, v in word_map.items()}

         with torch.no_grad():
             word_seq = caption_image(encoder, decoder, pil_img, word_map, beam_size=3)
             words = " ".join([rev_word_map[ind] for ind in word_seq if ind not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]) 
         st.image(pil_img, caption= words, width = 250)


def caption_image(encoder, decoder, img, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process

    trans = T.Compose(
        [
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = trans(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(
        k, num_pixels, encoder_dim
    )  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.tensor([[word_map["<start>"]]] * k, dtype=torch.long) # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1)  # (k, 1)

    # Lists to store completed sequences and corresponding scores
    complete_seqs = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, _ = decoder.attention(
            encoder_out, h
        )  # (s, encoder_dim), (s, num_pixels)


        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(
            torch.cat([embeddings, awe], dim=1), (h, c)
        )  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        # import pdb; pdb.set_trace()
        seqs = torch.cat(
            [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1
        )  # (s, step+1)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [
            ind
            for ind, next_word in enumerate(next_word_inds)
            if next_word != word_map["<end>"]
        ]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])

        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break

        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break

        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]

    return seq
