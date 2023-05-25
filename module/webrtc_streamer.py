
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from sample_utils.turn import get_ice_servers

def create_webrtc_streamer(*, key: str, video_frame_callback, queued_video_frames_callback = None):
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
    })
    WebrtcStreamerManager.add_ctx(key, ctx)
    return ctx

class WebrtcStreamerManager():
    _ctx_dict = {}
    _ctx_list = []

    @classmethod
    def get_ctx(cls, key: str):
        return cls._ctx_dict[key]
    
    @classmethod
    def add_ctx(cls, key: str, ctx):
        cls._ctx_dict[key] = ctx
        cls._ctx_list.append(ctx)
        print(cls._ctx_list)
