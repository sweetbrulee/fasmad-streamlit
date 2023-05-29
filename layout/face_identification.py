from typing_extensions import override
import av
import streamlit as st
from ._base import BaseLayout
from model.messagetuple import DetectionMetadata
from module.webrtc_streamer import create_webrtc_streamer

from service import FireDetection


class FaceIdentificationLayout(BaseLayout):
    @override
    def __init__(self):
        super().__init__()
        self.key = "face-identification"

    @override
    def mount(self):
        st.title("陌生人员监控")

        def callback(frame):
            img = frame.to_ndarray(format="bgr24")
            metadata_ret = FireDetection.create(img)
            # frame_ret = av.VideoFrame.from_ndarray(img, format="bgr24")

            # put into the queue
            self.metadata_queue_ref.put(
                [DetectionMetadata(boxes=metadata_ret, group=self.key)]
            )

            return frame

        self.video_frame_callback = callback

        self.webrtc_ctx = create_webrtc_streamer(self.webrtc_streamer_attributes)

    @override
    def update_impl(self):
        print("🥰🥰🥰🥰🥰🥰🥰🥰🥰")
