# -*- coding: utf-8 -*-
"""
Created on 2018/12/9 下午4:58

@author: Evan Chen
"""

from datetime import timedelta

from flask import Flask
from flask import render_template

app = Flask('人脸识别系统', template_folder='server/templates', static_folder='server/static')
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)

# 更改默认背景
import cv2, os

basepath = os.path.dirname(__file__)
dufault_path = os.path.join(basepath, 'static/images', 'leaves.png')
img = cv2.imread(dufault_path)

cv2.imwrite(os.path.join(basepath, 'static/images/show', 'first.jpg'), img)
cv2.imwrite(os.path.join(basepath, 'static/images/show', 'second.jpg'), img)
cv2.imwrite(os.path.join(basepath, 'static/images/show', 'third.jpg'), img)
cv2.imwrite(os.path.join(basepath, 'static/images/show', 'fourth.jpg'), img)

from server.views import home as home_blueprint
app.register_blueprint(home_blueprint)

@app.errorhandler(404)
def page_not_found(error):
    return render_template('fr_index.html'), 404
