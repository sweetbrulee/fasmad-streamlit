import streamlit as st
from streamlit_webrtc import WebRtcStreamerContext, webrtc_streamer
from model.webrtc_streamer_attributes import WebRTCStreamerAttributes
from module.webrtc_streamer import create_webrtc_streamer


def use_experimental_rerun_button():
    return st.button("重新运行 (实验性)", on_click=lambda: st.experimental_rerun())


'''
def use_recreate_webrtc_context_button(
    ctx_holder: WebRtcStreamerContext,
    webrtc_streamer_attributes: WebRTCStreamerAttributes,
):
    """This is useful when you want to recreate a webrtc context for the reason that the TURN server connection has expired. Note that this will also clear all the cached data so be careful to use this.

    Args:
        ctx_holder (object): The object that holds the webrtc context.
        webrtc_streamer_attributes (WebRTCStreamerAttributes): The webrtc streamer attributes type object, you can find it at ```model.WebRTCStreamerAttributes```.

    Returns:
        bool: The button component.
    """

    def recreate_webrtc_context():
        nonlocal ctx_holder
        st.cache_data.clear()  # This might be a dangerous operation.

        ctx_holder = create_webrtc_streamer(
            webrtc_streamer_attributes
        )  # re-assign the webrtc context to the holder

    return st.button("重新创建 WebRTC 连接 (不建议)", on_click=recreate_webrtc_context)
'''
