from __future__ import annotations
from typing import Callable, Optional
from dataclasses import dataclass

from aiortc.mediastreams import MediaStreamTrack
from streamlit_webrtc import Translations, WebRtcMode


@dataclass
class WebRtcStreamerAttributes:
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
    desired_playing_state: bool | None = None
    source_video_track: MediaStreamTrack | None = None
