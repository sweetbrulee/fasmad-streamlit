from streamlit_webrtc import Translations, WebRtcMode, webrtc_streamer
from model.webrtc_streamer_attributes import WebRtcStreamerAttributes
from sample_utils.turn import get_ice_servers

COMMON_RTC_CONFIG = {"iceServers": get_ice_servers()}


def create_webrtc_streamer(webrtc_streamer_attributes: WebRtcStreamerAttributes):
    def if_valid_then_return(value, default_value):
        return default_value if value is None else value

    attr = webrtc_streamer_attributes

    ctx = webrtc_streamer(
        key=attr.key,
        mode=if_valid_then_return(attr.mode, WebRtcMode.SENDRECV),
        rtc_configuration=if_valid_then_return(
            attr.rtc_configuration, COMMON_RTC_CONFIG
        ),
        video_frame_callback=if_valid_then_return(attr.video_frame_callback, None),
        queued_video_frames_callback=if_valid_then_return(
            attr.queued_video_frames_callback, None
        ),
        media_stream_constraints=if_valid_then_return(
            attr.media_stream_constraints, {"video": True, "audio": False}
        ),
        async_processing=if_valid_then_return(attr.async_processing, True),
        translations=if_valid_then_return(
            attr.translations,
            Translations(
                start="开始监测",
                stop="停止监测",
                select_device="选择摄像头",
                media_api_not_available="媒体 API 不可用",
                device_ask_permission="请允许使用摄像头",
                device_not_available="摄像头不可用",
                device_access_denied="摄像头访问被拒绝",
            ),
        ),
    )

    return ctx
