from streamlit_webrtc import WebRtcMode, webrtc_streamer

from sample_utils.turn import get_ice_servers


def create_webrtc_streamer(
    *, key: str, video_frame_callback, queued_video_frames_callback
):
    ctx = webrtc_streamer(
        key=key,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": get_ice_servers()},
        video_frame_callback=video_frame_callback,
        queued_video_frames_callback=queued_video_frames_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        translations={
            "start": "开始监测",
            "stop": "停止监测",
            "select_device": "选择摄像头",
            "media_api_not_available": "媒体 API 不可用",
            "device_ask_permission": "请允许使用摄像头",
            "device_not_available": "摄像头不可用",
            "device_access_denied": "摄像头访问被拒绝",
        },
    )
    return ctx
