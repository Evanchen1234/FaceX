# -*- coding: utf-8 -*-
"""
Created on 2018/12/9 下午5:01

@author: Evan Chen
"""

import os

from flask import render_template, request, jsonify

from model import set_png_for_show, LinearRegressionClassifier
from . import home

basepath = os.path.dirname(os.path.dirname(__file__))
lr = LinearRegressionClassifier()


@home.route('/', methods=['GET', 'POST'])
def home_page():
    return render_template('fr_index.html')


@home.route('/upload', methods=['POST'])
def upload():
    image = request.files['image']

    upload_path = os.path.join(basepath, 'static/images', 'detected.png')
    image.save(upload_path)

    return jsonify({'status': 'upload_success'})


@home.route('/detected', methods=['GET'])
def detected():
    detected_path = os.path.join(basepath, 'static/images', 'detected.png')
    preds = lr.predict(detected_path)

    # 从训练集合找寻相应的图像并显示
    set_png_for_show(preds)
    return render_template('fr_index.html')
