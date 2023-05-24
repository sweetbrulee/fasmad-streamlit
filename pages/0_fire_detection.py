import logging
import queue
from pathlib import Path
from typing import List, NamedTuple

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from sample_utils.download import download_file
from sample_utils.turn import get_ice_servers

from my_Fire_Detection.interface import start_fire_detect2

st.title("My first Streamlit app")
st.write("Hello, world")


def callback(frame):
    img = frame.to_ndarray(format="bgr24")

    im0, results = start_fire_detect2(img)

    return av.VideoFrame.from_ndarray(im0, format="bgr24")


webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": get_ice_servers()},
    video_frame_callback=callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
