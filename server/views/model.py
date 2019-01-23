# -*- coding: utf-8 -*-
"""
Created on 2019/1/21 下午7:29

@author: Evan Chen
"""

import os
from PIL import Image
import numpy as np
import pandas as pd
import shutil
from sklearn.decomposition import PCA
from sklearn.externals import joblib

from numpy import dot
from numpy.linalg import norm, pinv

basepath = os.path.dirname(os.path.dirname(__file__))

train_folder = os.path.join(basepath, 'static/images/train')
test_folder = os.path.join(basepath, 'static/images/test')

def generate_train_test(image_folder, n_remain=7):
    """
    划分训练集和测试集, 并保存到本地
    """
    image_data = []
    for path in os.walk(image_folder):
        if len(path[2]) == 0:
            continue

        for png in path[2]:
            if not png.endswith('png'): continue
            png_path = os.path.join(path[0], png)
            data = png_path.split('/')[-2:]
            data.append('_'.join(data))
            data.append(png_path)
            image_data.append(data)
    image_data = pd.DataFrame(image_data)

    # 对样本按照不同的人进行分层抽样
    # n_remain代表每个人保留的图像的张数，用于训练模型
    train_index, test_index = [], []
    for label in image_data[0].unique():
        df = image_data[image_data[0] == label]
        train_index += df.sample(n_remain, random_state=1024).index.tolist()
    test_index = list(set(image_data.index.tolist()) - set(train_index))

    # 将抽样结果写入本地的 train文件夹 与 test文件夹
    train_path_ls = image_data.iloc[train_index][3].tolist()
    test_path_ls = image_data.iloc[test_index][3].tolist()

    train_copy_path = image_data.iloc[train_index][2].apply(
        lambda x: train_folder + x.replace('.png', '_train.png')).tolist()
    test_copy_path = image_data.iloc[test_index][2].apply(
        lambda x: test_folder + x.replace('.png', '_test.png')).tolist()

    for old, new in zip(train_path_ls + test_path_ls, train_copy_path + test_copy_path):
        shutil.copy(old, new)


def image2vec(image_path, is_train=True):
    """
    将图像转化为特征向量+路径的形式
    当用于训练时，image_path为训练集文件目录
    当用于预测时，image_path为文件具体路径
    """
    image_data = []
    if is_train:
        for path in os.walk(image_path):
            for png in path[2]:
                if not png.endswith('png'): continue
                i_path = os.path.join(path[0], png)
                new_data = read_single_image(i_path)
                new_data.append(i_path)
                image_data.append(new_data)
    else:
        new_data = read_single_image(image_path)
        new_data.append(image_path)
        image_data.append(new_data)

    return pd.DataFrame(image_data)


def read_single_image(image_path):
    im = Image.open(image_path)
    width, height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data, dtype="float")  # /255.0
    new_data = np.reshape(data, (1, height * width)).tolist()[0]
    return new_data


def fit_save_pca(size=200):
    """
    根据训练集样本训练并保存pca模型
    """
    image_data = image2vec(train_folder, is_train=True)
    # 删除保存的路径信息
    norm_data = image_data.drop(10304, axis=1)
    # 做pca之前进行z-score操作
    norm_data = (norm_data - norm_data.mean()) / norm_data.std()

    # 训练pca模型
    pca = PCA(n_components=size, copy=True, svd_solver='auto')
    pca.fit(norm_data.values)

    # 保存模型
    pca_path = os.path.join(basepath, 'static/model', 'pca.m')
    joblib.dump(pca, pca_path)


def set_png_for_show(preds):
    """
    根据预测的结果，从训练集合中选出相同类型的图像显示
    """
    png = None
    for path in os.walk(train_folder):
        for item in path[2]:
            if preds[0] in item:
                png = item


    # 设置显示预测图片
    detected = os.path.join(train_folder, png)
    second = os.path.join(basepath, 'static/images/show', 'second.jpg')
    shutil.copy(detected, second)

    # 设置显示上传的图像
    original = os.path.join(basepath, 'static/images', 'detected.png')
    first = os.path.join(basepath, 'static/images/show', 'first.jpg')
    shutil.copy(original, first)

class LinearRegressionClassifier():
    def __init__(self):
        image_data = image2vec(train_folder, is_train=True)
        train = image_data.drop(10304, axis=1)
        self.label = image_data[10304].apply(lambda x: x.split('/')[-1].split('_')[0])

        pca_path = os.path.join(basepath, 'static/model', 'pca.m')
        self.pca = joblib.load(pca_path)
        self.A = self.pca.transform(train.values)

    def predict(self, detected_path):
        test_data = image2vec(detected_path, is_train=False).drop(10304, axis=1)
        y = self.pca.transform(test_data.values)
        weight = dot(dot(y, self.A.T), pinv(dot(self.A, self.A.T)))

        weight_df = pd.DataFrame(list(zip(weight[0], self.label.tolist())))
        min_dist = 10000
        preds = None
        for label in weight_df[1].unique():
            models = weight_df[weight_df[1] == label]

            train_data = self.A[models.index]
            local_weights = models[0].values.reshape(1, 7)

            residual = dot(local_weights, train_data) - y
            dist = norm(residual, 2)

            if dist < min_dist:
                min_dist = dist
                preds = label

        return [preds] if preds is not None else ['拒绝识别']