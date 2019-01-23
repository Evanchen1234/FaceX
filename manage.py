# -*- coding: utf-8 -*-
"""
Created on 2018/12/9 下午4:58

@author: Evan Chen
"""

from server import app

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
