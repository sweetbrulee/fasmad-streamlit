import logging
import threading

import av
import cv2
import streamlit as st
from matplotlib import pyplot as plt
from module.webrtc_streamer import create_webrtc_streamer


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

    # process results into metadata
    # and -> metadata_queue.put(processed_results)

    with lock:
        image_container["img"] = img_ret
    with lock:
        metadata_container["metadata"] = metadata_ret

    return frame_ret


ctx = create_webrtc_streamer(
    key="fire-detection",
    video_frame_callback=callback,
    queued_video_frames_callback=None,
)

fig_place = st.empty()
metadata_place = st.empty()
fig, ax = plt.subplots(1, 1)


# During media streaming, while the callback continues to be called,
# the main script execution stops at the bottom as usual.
# So we need to use a loop to keep the main script running and get the values from the callback in the outer scope.
while ctx.state.playing:
    with lock:
        img = image_container["img"]
    with lock:
        metadata_place.text(str(metadata_container["metadata"]))
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ax.cla()
    ax.hist(gray.ravel(), 256, [0, 256])
    fig_place.pyplot(fig)
