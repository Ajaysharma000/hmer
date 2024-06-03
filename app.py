import streamlit as st
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torch.nn.functional as F
from numpy import array
import matplotlib.pyplot as plt

# Assuming these modules are available and correctly implemented
from data_iterator import dataIterator
from Attention_RNN import AttnDecoderRNN
from Densenet_torchvision import densenet121

gpu = [0]
dictionaries = ['dictionary.txt']
hidden_size = 256
batch_size_t = 1
maxlen = 100


@st.cache_data
def load_dict(dictFile):
    with open(dictFile) as fp:
        stuff = fp.readlines()
    lexicon = {w[0]: int(w[1]) for w in (l.strip().split() for l in stuff)}
    return lexicon


worddicts = load_dict(dictionaries[0])
worddicts_r = [None] * len(worddicts)
for kk, vv in worddicts.items():
    worddicts_r[vv] = kk


def for_test(x_t):
    h_mask_t = []
    w_mask_t = []
    encoder = densenet121()
    attn_decoder1 = AttnDecoderRNN(hidden_size, 112, dropout_p=0.5)

    encoder = torch.nn.DataParallel(encoder, device_ids=gpu)
    attn_decoder1 = torch.nn.DataParallel(attn_decoder1, device_ids=gpu)
    encoder = encoder.cuda()
    attn_decoder1 = attn_decoder1.cuda()

    encoder.load_state_dict(torch.load('encoder_lr0.00001_BN_te1_d05_SGD_bs8_mask_conv_bn_b.pkl'))
    attn_decoder1.load_state_dict(torch.load('attn_decoder_lr0.00001_BN_te1_d05_SGD_bs8_mask_conv_bn_b.pkl'))

    encoder.eval()
    attn_decoder1.eval()

    x_t = Variable(x_t.cuda())
    x_mask = torch.ones(x_t.size()).cuda()
    x_t = torch.cat((x_t, x_mask), dim=1)
    x_real_high = x_t.size()[2]
    x_real_width = x_t.size()[3]
    h_mask_t.append(int(x_real_high))
    w_mask_t.append(int(x_real_width))
    x_real = x_t[0][0].view(x_real_high, x_real_width)
    output_highfeature_t = encoder(x_t)

    x_mean_t = torch.mean(output_highfeature_t).item()
    output_area_t1 = output_highfeature_t.size()
    output_area_t = output_area_t1[3]
    dense_input = output_area_t1[2]

    decoder_input_t = torch.LongTensor([111] * batch_size_t).cuda()
    decoder_hidden_t = torch.randn(batch_size_t, 1, hidden_size).cuda() * x_mean_t
    decoder_hidden_t = torch.tanh(decoder_hidden_t)

    prediction = torch.zeros(batch_size_t, maxlen)
    decoder_attention_t_cat = []

    for i in range(maxlen):
        decoder_output, decoder_hidden_t, decoder_attention_t, attention_sum_t = attn_decoder1(
            decoder_input_t, decoder_hidden_t, output_highfeature_t, output_area_t, None, None, dense_input,
            batch_size_t, h_mask_t, w_mask_t, gpu)

        decoder_attention_t_cat.append(decoder_attention_t[0].data.cpu().numpy())
        topv, topi = torch.max(decoder_output, 2)
        if torch.sum(topi) == 0:
            break
        decoder_input_t = topi.view(batch_size_t)

        prediction[:, i] = decoder_input_t

    k = np.array(decoder_attention_t_cat)
    x_real = np.array(x_real.cpu().data)

    prediction = prediction[0]
    prediction_real = []
    for ir in range(len(prediction)):
        if int(prediction[ir]) == 0:
            break
        prediction_real.append(worddicts_r[int(prediction[ir])])
    prediction_real.append('<eol>')

    prediction_real_show = np.array(prediction_real)
    return k, prediction_real_show


st.title('HMER Tool V2.0')

uploaded_file = st.file_uploader("Choose an image...", type="bmp")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    img_open2 = torch.from_numpy(np.array(image)).type(torch.FloatTensor) / 255.0
    img_open2 = img_open2.unsqueeze(0).unsqueeze(0)

    attention, prediction = for_test(img_open2)

    prediction_string = ''.join([pred for pred in prediction if pred != '<eol>'])
    st.write(f"Prediction: {prediction_string}")

    for i in range(attention.shape[0]):
        if prediction[i] == '<eol>':
            continue
        attention2 = (attention[i, 0, :, :] * attention[i, 0, :, :])
        image_attention = np.array(image) + (attention2 * 1000)
        st.image(image_attention, caption=f'Attention map {i}', use_column_width=True)

        # Optional: Add matplotlib plot for better visualization
        fig, ax = plt.subplots()
        ax.imshow(image_attention, cmap='hot')
        ax.set_title(f'Attention map {i}')
        st.pyplot(fig)
