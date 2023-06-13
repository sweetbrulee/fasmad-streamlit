import logging
import threading

import streamlit as st
from layout import FaceIdentificationLayout, FireDetectionLayout
from module.developer import (
    mount_experimental_rerun_button,
    mount_clear_all_cache_button,
)
from service.message_queue import MetadataQueueService
from module.alarm_agent import FaceAlarmAgent, FireAlarmAgent


logger = logging.getLogger(__name__)

lock = threading.Lock()  # for thread-safety

metadata_queue = MetadataQueueService.use_queue()

fire_alarm_agent = FireAlarmAgent(group="fire-detection")
face_alarm_agent = FaceAlarmAgent(group="face-identification")

face_layout, fire_layout = FaceIdentificationLayout(), FireDetectionLayout()

face_layout.mount()
fire_layout.mount()

if any([face_layout.streaming, fire_layout.streaming]):
    metadata_placeholder = None

    if fire_layout.alarm_placeholder:
        fire_alarm_agent.bind_container_callfunc(
            fire_layout.alarm_placeholder.container
        )
    if face_layout.alarm_placeholder:
        face_alarm_agent.bind_container_callfunc(
            face_layout.alarm_placeholder.container
        )

    fire_alarm_agent.reset_timers()
    face_alarm_agent.reset_timers()

    if st.checkbox("识别数据", value=False):
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

        fire_alarm_agent.run(group, boxes)
        face_alarm_agent.run(group, boxes)

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
