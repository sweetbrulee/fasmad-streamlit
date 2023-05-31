# Copyright (C) 2018-2021 coneypo
# SPDX-License-Identifier: MIT

# Author:   coneypo
# Blog:     http://www.cnblogs.com/AdaminXie
# GitHub:   https://github.com/coneypo/Dlib_face_recognition_from_camera
# Mail:     coneypo@foxmail.com

# 从人脸图像文件中提取人脸特征存入 "features_all.csv" / Extract features from images and save into "features_all.csv"

import os
import dlib
import csv
import numpy as np
import logging
import pandas as pd
import cv2

# 要读取人脸图像文件的路径 / Path of cropped faces
path_images_from_camera = "data/data_faces_from_camera/"

# Dlib 正向人脸检测器 / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib 人脸 landmark 特征点检测器 / Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet 人脸识别模型，提取 128D 的特征矢量 / Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


# 返回单张图像的 128D 特征 / Return 128D features for single image
# Input:    path_img           <class 'str'>
# Output:   face_descriptor    <class 'dlib.vector'>
def return_128d_features(img_rd):
    faces = detector(img_rd, 1)

    # 因为有可能截下来的人脸再去检测，检测不出来人脸了, 所以要确保是 检测到人脸的人脸图像拿去算特征
    # For photos of faces saved, we need to make sure that we can detect faces from the cropped images
    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    else:
        face_descriptor = 0
        logging.warning("no face")
    return face_descriptor


# 返回 personX 的 128D 特征均值 / Return the mean value of 128D face descriptor for person X
# Input:    path_face_personX        <class 'str'>
# Output:   features_mean_personX    <class 'numpy.ndarray'>
def return_features_mean_personX(img_rd_set):
    features_list_personX = []

    for img_rd in img_rd_set:
        # 调用 return_128d_features() 得到 128D 特征 / Get 128D features for single image of personX
        features_128d = return_128d_features(img_rd)
        # 遇到没有检测出人脸的图片跳过 / Jump if no face detected from image
        if features_128d != 0:
            features_list_personX.append(features_128d)

    # 计算 128D 特征的均值 / Compute the mean
    # personX 的 N 张图像 x 128D -> 1 x 128D
    if features_list_personX:
        features_mean_personX = np.array(features_list_personX, dtype=object).mean(axis=0)
    else:
        features_mean_personX = np.zeros(128, dtype=object, order='C')
    return features_mean_personX


def main(img_rd_set,person_name):
    logging.basicConfig(level=logging.INFO)
    path_features_known_csv = "data/features_all.csv"
    csv_rd = pd.read_csv(path_features_known_csv, header=None)
    features_mean_personX = return_features_mean_personX(img_rd_set)

    index = csv_rd.shape[0]
    for i in range(csv_rd.shape[0]):
        if person_name == csv_rd.iloc[i][0]:
            index = i
            for j in range(1,csv_rd.shape[1]):
                csv_rd.iloc[i,j] = 0.5*csv_rd.iloc[i][j] + 0.5*features_mean_personX[j-1]
            break
    if index == csv_rd.shape[0]:
        # features_mean_personX.insert(0,person_name)
        features_mean_personX = np.insert(features_mean_personX, 0, person_name, axis=0)
        csv_rd.loc[index] = features_mean_personX
    with open("data/features_all.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Get the mean/average features of face/personX, it will be a list with a length of 128D
        for index in range(csv_rd.shape[0]):
            writer.writerow(csv_rd.iloc[index])
            logging.info('\n')
        logging.info("%s的人脸数据存入 / Save the features of face registered into: data/features_all.csv",person_name)


if __name__ == '__main__':
    path1 = "data/data_faces_from_camera/person_1_hwy/img_face_1.jpg"
    img1 = cv2.imread(path1)

    img_rd_set = [img1]
    person_name = "balabala"
    main(img_rd_set,person_name)