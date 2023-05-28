import logging
import threading

import av
import streamlit as st
from model.messagetuple import DetectionMetadata
from model.webrtc_streamer_attributes import WebRTCStreamerAttributes
from module.webrtc_streamer import create_webrtc_streamer
from module.developer import use_experimental_rerun_button


from service import FireDetection, MetadataQueueService

st.title("火灾识别监控系统")

logger = logging.getLogger(__name__)

lock = threading.Lock()  # for thread-safety

metadata_queue = MetadataQueueService.use_queue()

image_container = {"img": None}
metadata_container = {"metadata": None}


def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img_ret, metadata_ret = FireDetection.create(img)
    frame_ret = av.VideoFrame.from_ndarray(img_ret, format="bgr24")

    # put into the queue
    metadata_queue.put([DetectionMetadata(boxes=metadata_ret)])

    return frame_ret


WEBRTC_STREAMER_ATTR = WebRTCStreamerAttributes(
    key="fire-detection",
    video_frame_callback=callback,
    queued_video_frames_callback=None,
)

webrtc_ctx = create_webrtc_streamer(WEBRTC_STREAMER_ATTR)


if webrtc_ctx.state.playing:
    metadata_placeholder = None
    if st.checkbox("识别结果", value=True):
        metadata_placeholder = st.empty()

    # NOTE: The video transformation with fire detection and
    # this loop displaying the result metadata are running
    # in different threads asynchronously.
    # Then the rendered video frames and the metadata displayed here
    # are not strictly synchronized.
    while True:
        boxes = metadata_queue.get()[0].boxes
        if metadata_placeholder is None:
            continue
        with metadata_placeholder.container():
            st.write(boxes)


# ------------------------------------------------------------\
# ------------------------------------------------------------|
# The following code is for developer use.                    |
# ------------------------------------------------------------|
# ------------------------------------------------------------/

if st.checkbox("开发者选项"):
    st.subheader("开发者选项")
    st.write("这些选项仅供开发者使用。")

    if use_experimental_rerun_button():
        pass
