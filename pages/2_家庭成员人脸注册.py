import logging
import streamlit as st
import cv2
import numpy as np

from module.developer import (
    mount_experimental_rerun_button,
    mount_clear_all_cache_button,
)
from service import FaceIdentification

logger = logging.getLogger(__name__)

pictures = []

img_file_buffers = [
    st.camera_input(
        "### 请将人脸置于框内，点击拍照按钮",
        key=f"facial_register_camera_input_{i}",
    )
    for i in range(5)
]

for img_file_buffer in img_file_buffers:
    if img_file_buffer:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        # Store into pictures list
        pictures.append(cv2_img)

name = st.text_input("请输入姓名")

if st.button("提交"):
    FaceIdentification.register(name, pictures)

st.image(pictures, channels="BGR", width=100, output_format="JPEG")


# ------------------------------------------------------------\
# ------------------------------------------------------------|
# The following code is for developer use.                    |
# ------------------------------------------------------------|
# ------------------------------------------------------------/

if st.checkbox("开发者选项", value=False):
    st.subheader("开发者选项")
    st.write("这些选项仅供开发者使用。")

    mount_experimental_rerun_button()
    mount_clear_all_cache_button()
