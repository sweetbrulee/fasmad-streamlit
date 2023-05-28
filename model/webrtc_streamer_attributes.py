from __future__ import annotations
from typing import Callable
from dataclasses import dataclass

from streamlit_webrtc import Translations, WebRtcMode


@dataclass
class WebRTCStreamerAttributes:
    """
    WebRTCStreamerAttributes is a data class that contains the attributes for the
    WebRTC streamer model.
    """

    key: str
    rtc_configuration: dict | None = None
    video_frame_callback: Callable | None = None
    queued_video_frames_callback: Callable | None = None
    media_stream_constraints: dict | None = None
    mode: WebRtcMode | None = None
    async_processing: bool | None = None
    translations: Translations | None = None
