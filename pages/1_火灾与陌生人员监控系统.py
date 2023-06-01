import logging
import threading

import streamlit as st
from layout import FaceIdentificationLayout, FireDetectionLayout
from module.developer import (
    mount_experimental_rerun_button,
    mount_clear_all_cache_button,
)
from service.message_queue import MetadataQueueService


logger = logging.getLogger(__name__)

lock = threading.Lock()  # for thread-safety

metadata_queue = MetadataQueueService.use_queue()

face_layout, fire_layout = FaceIdentificationLayout(), FireDetectionLayout()

face_layout.mount()
fire_layout.mount()
# mount_mixer_layout()

if any([face_layout.streaming, fire_layout.streaming]):
    metadata_placeholder = None
    if st.checkbox("识别结果", value=True):
        metadata_placeholder = st.empty()

    # NOTE: The video transformation with detection and
    # this loop displaying the result metadata are running
    # in different threads asynchronously.
    # Then the rendered video frames and the metadata displayed here
    # are not strictly synchronized.
    while True:
        fire_layout.update()
        face_layout.update()

        group, boxes = metadata_queue.get()[0]
        if metadata_placeholder is None:
            continue
        with metadata_placeholder.container():
            st.write(f"组别: {group}")
            st.write(boxes)

# ------------------------------------------------------------\
# ------------------------------------------------------------|
# The following code is for developer use.                    |
# ------------------------------------------------------------|
# ------------------------------------------------------------/

if st.checkbox("开发者选项", value=False):
    st.subheader("开发者选项")
    st.write("这些选项仅供开发者使用。")

    mount_experimental_rerun_button()
    mount_clear_all_cache_button()
