from typing_extensions import override
import av
import streamlit as st
from ._base import BaseLayout
from model.messagetuple import DetectionMetadata
from module.webrtc_streamer import create_webrtc_streamer

from service import FaceIdentification


class FaceIdentificationLayout(BaseLayout):
    @override
    def __init__(self):
        super().__init__()
        self.key = "face-identification"

    @override
    def mount(self):
        st.title("陌生人员监控")

        async def queued_callback(frames: list):
            frame = frames[0]
            img = frame.to_ndarray(format="bgr24")
            ret = FaceIdentification.create(img)
            metadata_ret = ret[0]
            img_ret = ret[1]
            # frame_ret = av.VideoFrame.from_ndarray(img_ret, format="bgr24")

            # put into the queue
            self.metadata_queue_ref.put(
                [DetectionMetadata(boxes=metadata_ret, group=self.key)]
            )

            return frames

        self.queued_video_frames_callback = queued_callback

        self.webrtc_ctx = create_webrtc_streamer(self.webrtc_streamer_attributes)

    @override
    def update(self):
        pass
