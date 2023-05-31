# Copyright (C) 2018-2021 coneypo
# SPDX-License-Identifier: MIT

# Author:   coneypo
# Blog:     http://www.cnblogs.com/AdaminXie
# GitHub:   https://github.com/coneypo/Dlib_face_recognition_from_camera
# Mail:     coneypo@foxmail.com

# 摄像头实时人脸识别 / Real-time face detection and recognition

import dlib
import numpy as np
import cv2
import pandas as pd
import os
import time
import logging
from PIL import Image, ImageDraw, ImageFont

# Dlib 正向人脸检测器 / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib 人脸 landmark 特征点检测器 / Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet 人脸识别模型，提取 128D 的特征矢量 / Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

path = "data/data_faces_from_camera/person_1_hwy/img_face_1.jpg"


class Face_Recognizer:
    def __init__(self):
        self.face_feature_known_list = []                # 用来存放所有录入人脸特征的数组 / Save the features of faces in database
        self.face_name_known_list = []                   # 存储录入人脸名字 / Save the name of faces in database

        self.current_frame_face_cnt = 0                     # 存储当前摄像头中捕获到的人脸数 / Counter for faces in current frame
        self.current_frame_face_feature_list = []           # 存储当前摄像头中捕获到的人脸特征 / Features of faces in current frame
        self.current_frame_face_name_list = []              # 存储当前摄像头中捕获到的所有人脸的名字 / Names of faces in current frame
        self.current_frame_face_name_position_list = []     # 存储当前摄像头中捕获到的所有人脸的名字坐标 / Positions of faces in current frame

        # Update FPS
        self.fps = 0                    # FPS of current frame
        self.fps_show = 0               # FPS per second
        self.frame_start_time = 0
        self.frame_cnt = 0
        self.start_time = time.time()

        self.font = cv2.FONT_ITALIC
        self.font_chinese = ImageFont.truetype("simsun.ttc", 30)

    # 从 "features_all.csv" 读取录入人脸特征 / Read known faces from "features_all.csv"
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_feature_known_list.append(features_someone_arr)
            logging.info("Faces in Database：%d", len(self.face_feature_known_list))
            return 1
        else:
            logging.warning("'features_all.csv' not found!")
            logging.warning("Please run 'get_faces_from_camera.py' "
                            "and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'")
            return 0

    # 计算两个128D向量间的欧式距离 / Compute the e-distance between two 128D features
    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist


    # 生成的 cv2 window 上面添加说明文字 / PutText on cv2 window
    # def draw_note(self, img_rd):
    #     cv2.putText(img_rd, "Face Recognizer", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    #     cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1,
    #                 cv2.LINE_AA)
    #     cv2.putText(img_rd, "FPS:    " + str(self.fps_show.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1,
    #                 cv2.LINE_AA)
    #     cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1,
    #                 cv2.LINE_AA)
    #     cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_name(self, img_rd):
        # 在人脸框下面写人脸名字 / Write names under rectangle
        img = Image.fromarray(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for i in range(self.current_frame_face_cnt):
            # cv2.putText(img_rd, self.current_frame_face_name_list[i], self.current_frame_face_name_position_list[i], self.font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
            draw.text(xy=self.current_frame_face_name_position_list[i], text=self.current_frame_face_name_list[i], font=self.font_chinese,
                  fill=(255, 255, 0))
            img_rd = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img_rd

    # 修改显示人名 / Show names in chinese
    def show_chinese_name(self):
        # Default known name: person_1, person_2, person_3
        if self.current_frame_face_cnt >= 1:
            # 修改录入的人脸姓名 / Modify names in face_name_known_list to chinese name
            self.face_name_known_list[0] = '张三'.encode('utf-8').decode()
            # self.face_name_known_list[1] = '张四'.encode('utf-8').decode()

    # 处理获取的视频流，进行人脸识别 / Face detection and recognition from input video stream
    def process(self, img_rd):
        # 1. 读取存放所有人脸特征的 csv / Read known faces from "features.all.csv"
        if self.get_face_database():
            faces = detector(img_rd, 0)

            # self.draw_note(img_rd)
            self.current_frame_face_feature_list = []
            self.current_frame_face_cnt = 0
            self.current_frame_face_name_position_list = []
            self.current_frame_face_name_list = []

            # 2. 检测到人脸 / Face detected in current frame
            if len(faces) != 0:
                # 3. 获取当前捕获到的图像的所有人脸的特征 / Compute the face descriptors for faces in current frame
                for i in range(len(faces)):
                    shape = predictor(img_rd, faces[i])
                    self.current_frame_face_feature_list.append(face_reco_model.compute_face_descriptor(img_rd, shape))
                # 4. 遍历捕获到的图像中所有的人脸 / Traversal all the faces in the database
                for k in range(len(faces)):
                    logging.debug("For face %d in camera:", k+1)
                    # 先默认所有人不认识，是 unknown / Set the default names of faces with "unknown"
                    self.current_frame_face_name_list.append("unknown")

                    # 每个捕获人脸的名字坐标 / Positions of faces captured
                    self.current_frame_face_name_position_list.append(tuple(
                        [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                    # 5. 对于某张人脸，遍历所有存储的人脸特征
                    # For every faces detected, compare the faces in the database
                    current_frame_e_distance_list = []
                    for i in range(len(self.face_feature_known_list)):
                        # 如果 person_X 数据不为空
                        if str(self.face_feature_known_list[i][0]) != '0.0':
                            e_distance_tmp = self.return_euclidean_distance(self.current_frame_face_feature_list[k],
                                                                            self.face_feature_known_list[i])
                            logging.debug("  With person %s, the e-distance is %f", str(i + 1), e_distance_tmp)
                            current_frame_e_distance_list.append(e_distance_tmp)
                        else:
                            # 空数据 person_X
                            current_frame_e_distance_list.append(999999999)
                    # 6. 寻找出最小的欧式距离匹配 / Find the one with minimum e-distance
                    similar_person_num = current_frame_e_distance_list.index(min(current_frame_e_distance_list))
                    logging.debug("Minimum e-distance with %s: %f", self.face_name_known_list[similar_person_num], min(current_frame_e_distance_list))

                    if min(current_frame_e_distance_list) < 0.4:
                        self.current_frame_face_name_list[k] = self.face_name_known_list[similar_person_num]
                        logging.debug("Face recognition result: %s", self.face_name_known_list[similar_person_num])
                    else:
                        logging.debug("Face recognition result: Unknown person")
                    logging.debug("\n")

                    # 矩形框 / Draw rectangle
                    for kk, d in enumerate(faces):
                        # 绘制矩形框
                        cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]),
                                      (255, 255, 255), 2)

                self.current_frame_face_cnt = len(faces)

                # 7. 在这里更改显示的人名 / Modify name if needed
                # self.show_chinese_name()

                # 8. 写名字 / Draw name
                img_with_name = self.draw_name(img_rd)

            else:
                img_with_name = img_rd

            logging.debug("Faces in camera now: %s", self.current_frame_face_name_list)

            return img_with_name
            # cv2.imshow("camera", img_with_name)
            # k = cv2.waitKey(0)  # waitKey代表读取键盘的输入，0代表一直等待
            # if k == 27:  # 键盘上Esc键的键值
            #     cv2.destroyAllWindows()

    def run(self,img_rd):
        # img_rd = cv2.imread(path)
        # print("1111111111\n")
        result = self.process(img_rd)
        return [self.current_frame_face_name_list,result]
        # print("1111111111\n")


def main(img_rd):
    # logging.basicConfig(level=logging.DEBUG) # Set log level to 'logging.DEBUG' to print debug info of every frame
    logging.basicConfig(level=logging.INFO)
    Face_Recognizer_con = Face_Recognizer()
    return Face_Recognizer_con.run(img_rd)


if __name__ == '__main__':
    # img_rd = np.array([1,2,5,6,7])
    path1 = "data/data_faces_from_camera/person_1_hwy/img_face_1.jpg"
    img_rd = cv2.imread(path1)
    name_list,img_with_name = main(img_rd)
    print(name_list)
    cv2.imshow("camera", img_with_name)
    k = cv2.waitKey(0)  # waitKey代表读取键盘的输入，0代表一直等待
    if k == 27:  # 键盘上Esc键的键值
        cv2.destroyAllWindows()