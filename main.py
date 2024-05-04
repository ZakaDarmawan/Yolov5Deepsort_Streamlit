import streamlit as st
from track import *
import tempfile
import cv2
import torch
import os
import numpy as np
import matplotlib as plt
from urllib.parse import urlparse

if __name__ == '__main__':
    
    st.sidebar.header('🚨 Konfigurasi Model')
    # custom class
    assigned_class_id = [0, 1, 2, 3]
    names = ['bus', 'mobil', 'motor', 'truk']

    # Always display the multiselect widget for selecting custom classes
    assigned_class_id = []
    assigned_class = st.sidebar.multiselect('Pilih class spesifik untuk proses deteksi', names)
    for each in assigned_class:
        assigned_class_id.append(names.index(each))

    # Display selected class in the sidebar
    st.sidebar.caption("class terpilih : {}".format(', '.join(assigned_class) if assigned_class else 'None'))

    # st.write(assigned_class_id)
    # setting hyperparameter
    line = st.sidebar.number_input('🛠️ Line position', min_value=0.0, max_value=1.0, value=0.6, step=0.1)
    confidence = st.sidebar.slider('⛏️ Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.write("Confidence set :", confidence)
    iou_thres = st.sidebar.slider('✏️ Iou Threshold', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.write("Iou set :", iou_thres)

    st.subheader("🎦 Video Input")
    st.write("""silahkan mengunggah video dengan ketentuan tidak lebih dari 200MB untuk 
            mempercepat proses unggah dan proses deteksi. format video yang didukung yaitu MP4, MOV, AVI.""")
    # upload video
    video_file_buffer = st.file_uploader("Silahkan Upload Video Untuk Memulai Deteksi Objek", type=['mp4', 'mov', 'avi'])
    newpath = r"runs/video_upload"
    if not os.path.exists(newpath): os.makedirs(newpath) 
    
    if video_file_buffer:
        st.success("File Uploaded")
        st.text('"Detail video"')
        st.video(video_file_buffer)
        # save video from streamlit into "videos" folder for future detect
        with open(os.path.join("runs/video_upload", video_file_buffer.name), 'wb') as f:
            f.write(video_file_buffer.getbuffer())

    status = st.empty()
    stframe = st.empty()
    if video_file_buffer is None:
        status.markdown('<font size= "4"> **Status:** Waiting for input </font>', unsafe_allow_html=True)
    else:
        status.markdown('<font size= "4"> **Status:** Ready </font>', unsafe_allow_html=True)
   
    mobil, bus, truk, motor = st.columns(4)

    with mobil:
        st.markdown('**Mobil**')
        mobil_text = st.markdown('__')
    with bus:
        st.markdown('**Bus**')
        bus_text = st.markdown('__')
    with truk:
        st.markdown('**Truk**')
        truk_text = st.markdown('__')   
    with motor:
        st.markdown('**Motor**')
        motor_text = st.markdown('__')

    fps, _,  _, _  = st.columns(4)
    with fps:
        st.markdown('**FPS**')
    fps_text = st.markdown('__')
    
    col1, col2, col3 = st.columns([5, 5, 10], gap="small")
    with col1:
        track_button = st.button('Mulai Proses Video!')
    with col2:
        track_stop_button = st.button('Stop Proses Video!')
    # reset_button = st.sidebar.button('RESET ID')
    if track_button:
        # reset ID and count from 0
        reset()
        opt = parse_opt()
        opt.conf_thres = confidence
        opt.source = f'runs/video_upload/{video_file_buffer.name}'

        status.markdown('<font size= "4"> **Status:** Running... </font>', unsafe_allow_html=True)
        with torch.no_grad():
            detect(opt, stframe, mobil_text, bus_text, truk_text, motor_text, line, fps_text, assigned_class_id)
        status.markdown('<font size= "4"> **Status:** Finished ! </font>', unsafe_allow_html=True)
   
