import logging
from pathlib import Path

import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from sample_utils.download import download_file
from sample_utils.turn import get_ice_servers

from service import FireDetection, MetadataQueueService

st.title("火灾识别监控系统")

logger = logging.getLogger(__name__)

metadata_queue = MetadataQueueService.use_queue()


def callback(frame):
    img = frame.to_ndarray(format="bgr24")

    im0, results = FireDetection.create(img)

    # process results into metadata
    # and -> metadata_queue.put(processed_results)

    return av.VideoFrame.from_ndarray(im0, format="bgr24")


webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": get_ice_servers()},
    video_frame_callback=callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
