# -*- coding: utf-8 -*-
import sys
import uuid
import requests
import hashlib
import time
from imp import reload

import time

# reload(sys)
# 付费的有道翻译API，稳定性可靠性佳 #
class YoudaoAPI:
    __YOUDAO_URL = 'https://openapi.youdao.com/api'
    __APP_KEY = '22cb2d01411ab56d'  # 自己账号的ID
    __APP_SECRET = 'Rbo57nVgiAFxOPNsXkJ0HJAKdnWJq1gm' # 自己账号的秘钥

    def __encrypt(self, signStr):
        hash_algorithm = hashlib.sha256()
        hash_algorithm.update(signStr.encode('utf-8'))
        return hash_algorithm.hexdigest()


    def __truncate(self, q):
        if q is None:
            return None
        size = len(q)
        return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]


    def __do_request(self, data):
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        return requests.post(self.__YOUDAO_URL, data=data, headers=headers)


    def connect(self, q):
        data = {}
        data['from'] = 'AUTO'
        data['to'] = 'AUTO'
        data['signType'] = 'v3'
        curtime = str(int(time.time()))
        data['curtime'] = curtime
        salt = str(uuid.uuid1())
        signStr = self.__APP_KEY + self.__truncate(q) + salt + curtime + self.__APP_SECRET
        sign = self.__encrypt(signStr)
        data['appKey'] = self.__APP_KEY
        data['q'] = q
        data['salt'] = salt
        data['sign'] = sign
        response = self.__do_request(data)
        contentType = response.headers['Content-Type']
        if contentType == "audio/mp3":
            millis = int(round(time.time() * 1000))
            filePath = "合成的音频存储路径" + str(millis) + ".mp3"
            fo = open(filePath, 'wb')
            fo.write(response.content)
            fo.close()
        else:
            # print()
            return response.json()['translation'][-1]