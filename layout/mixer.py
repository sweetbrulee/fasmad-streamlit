import math
from typing import List

from model.webrtc_streamer_attributes import WebRtcStreamerAttributes
from module.webrtc_streamer import create_webrtc_streamer

import av
import cv2
import numpy as np
from streamlit_webrtc import (
    WebRtcMode,
    create_mix_track,
    create_process_track,
)

__all__ = ["mount_mixer_layout"]


def mount_mixer_layout(*webrtc_streamer_ctx):
    """Mount the mix output layout on the page."""

    # Terminology as of now:
    ## input_tracks: captured by devices
    ## output_tracks: being processed by AI models

    # For each WebRTC streamer context, create a process track,
    # process tracks will be the input tracks later on.
    process_tracks = [
        create_process_track(ctx.output_video_track) for ctx in webrtc_streamer_ctx
    ]

    # Create a mix track.
    mix_track = create_mix_track(kind="video", mixer_callback=mixer_callback, key="mix")

    # Create a mix context.
    mix_ctx = create_webrtc_streamer(
        WebRtcStreamerAttributes(
            key="mix",
            mode=WebRtcMode.RECVONLY,
            source_video_track=mix_track,
            desired_playing_state=any(
                (ctx.state.playing for ctx in webrtc_streamer_ctx)
            ),
        )
    )

    # Terminology as of now:
    ## input_tracks: process_tracks
    ## output_tracks: there's only one output track, which is mix_track

    # Add all process tracks to the mix_ctx source track as input tracks.
    for track in process_tracks:
        if mix_ctx.source_video_track and track:
            mix_ctx.source_video_track.add_input_track(track)


def mixer_callback(frames: List[av.VideoFrame]) -> av.VideoFrame:
    buf_w = 640
    buf_h = 480
    buffer = np.zeros((buf_h, buf_w, 3), dtype=np.uint8)

    n_inputs = len(frames)

    n_cols = math.ceil(math.sqrt(n_inputs))
    n_rows = math.ceil(n_inputs / n_cols)
    grid_w = buf_w // n_cols
    grid_h = buf_h // n_rows

    for i in range(n_inputs):
        frame = frames[i]
        if frame is None:
            continue

        grid_x = (i % n_cols) * grid_w
        grid_y = (i // n_cols) * grid_h

        img = frame.to_ndarray(format="bgr24")
        src_h, src_w = img.shape[0:2]

        aspect_ratio = src_w / src_h

        window_w = min(grid_w, int(grid_h * aspect_ratio))
        window_h = min(grid_h, int(window_w / aspect_ratio))

        window_offset_x = (grid_w - window_w) // 2
        window_offset_y = (grid_h - window_h) // 2

        window_x0 = grid_x + window_offset_x
        window_y0 = grid_y + window_offset_y
        window_x1 = window_x0 + window_w
        window_y1 = window_y0 + window_h

        buffer[window_y0:window_y1, window_x0:window_x1, :] = cv2.resize(
            img, (window_w, window_h)
        )

    new_frame = av.VideoFrame.from_ndarray(buffer, format="bgr24")

    return new_frame
