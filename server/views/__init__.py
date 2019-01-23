# -*- coding: utf-8 -*-
"""
Created on 2018/12/9 下午4:59

@author: Evan Chen
"""
from flask import Blueprint

home = Blueprint('home',__name__)

import server.views.router