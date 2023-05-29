from typing import final
import streamlit as st

from abc import ABC, abstractmethod

from model.webrtc_streamer_attributes import WebRtcStreamerAttributes
from service.message_queue import MetadataQueueService


class BaseLayout(ABC):
    @abstractmethod
    def __init__(self):
        self.image_container = {"img": None}
        self.metadata_container = {"metadata": None}
        self.metadata_queue_ref = MetadataQueueService.use_queue()
        self.metadata_placeholder_ref = None
        self.key = ""

        self.video_frame_callback = None
        self.queued_video_frames_callback = None
        self.webrtc_ctx = None
        self._webrtc_streamer_attributes = None

    @property
    def webrtc_streamer_attributes(self):
        return WebRtcStreamerAttributes(
            key=self.key,
            video_frame_callback=self.video_frame_callback,
            queued_video_frames_callback=self.queued_video_frames_callback,
        )

    @property
    def streaming(self):
        return self.webrtc_ctx.state.playing

    @abstractmethod
    def mount(self):
        pass

    @abstractmethod
    def update(self):
        """Update the layout UI state and re-render the layout."""
        pass
