import json

import requests
import random
from googletrans import Translator
from langdetect import detect
from nltk import tokenize
from numpy import size
from youdaoAPI import YoudaoAPI # 付费有道翻译API #

class Datastengthen:
    # 免费有道翻译API——1h1000次，稳定性差 #
    def __youdaotrans(self, query):
        url = 'http://fanyi.youdao.com/translate'
        data = {
            "i": query,  # 待翻译的字符串
            "from": "AUTO", # 语言类型自动是英汉互译
            "to": "AUTO",
            "smartresult": "dict",
            "client": "fanyideskweb",
            "salt": "16081210430980",
            "doctype": "json",
            "version": "2.1",
            "keyfrom": "fanyi.web",
            "action": "FY_BY_CLICKBUTTION"
        }
        res = requests.post(url, data=data).json()
        return res['translateResult'][0][0]['tgt']  # 返回翻译后的结果

# 使用有道词典API单句反向翻译，按比例随机 #
    def stengthbytranspart(self, filename, percentage):
        with open("data/"+filename, 'r', encoding="UTF-8") as f:
            # with open("data/train_translated_filtered_stopwords.json", 'r', encoding="UTF-8") as f:
            data = json.load(f)
        f.close()

        initsize = size(data)
        currplus = initsize - 2800  # 用于断点续传
        print(initsize)
        youdao = YoudaoAPI()
        for i in range(0, 2200):
            if data[i]['label'] == 0:
                if currplus > 0:
                    currplus -= 1
                    continue
                else:
                    print(i)
                    splitnews = tokenize.sent_tokenize(data[i]['content'])
                    # print(data[i]['content'])
                    newnews = ''
                    for sent in splitnews:
                        # print(sent)
                        if len("".join([char for char in sent if char != ' '])) < 5: # 避免传入空字符串引起报错
                            continue
                        elif random.random() < percentage:
                            newsent = youdao.connect(youdao.connect(sent))
                            newnews = newnews + newsent + '. '
                        else:
                            newnews = newnews + sent + '. '
                    print(data[i]['content'])
                    print(newnews)
                    # exit()
                    newitem = {}
                    print(i)
                    newitem['content'] = newnews
                    newitem['label'] = 0
                    data.append(newitem)

                    out = json.dumps(data, ensure_ascii=False)
                    f2 = open("data\\" + filename, 'w', encoding='UTF-8')
                    f2.write(out)
                    f2.close()
                    print("数据更新完成！")
        print("newsize{}".format(size(data)))

        out = json.dumps(data, ensure_ascii=False)
        f2 = open("data\\" + filename, 'w', encoding='UTF-8')
        f2.write(out)
        f2.close()
        print("数据增强完成！")

# 使用谷歌翻译API全文反向翻译 #
    def stengthbytranswhole(self, filename):
        googletranslator = Translator(service_urls=['translate.google.cn', 'translate.google.hk'])
        with open("data/"+filename, 'r', encoding="UTF-8") as f:
            data = json.load(f)
        f.close()

        initsize = size(data)
        currplus = initsize - 2200
        print(initsize)

        print(initsize)
        for index in range(0, 2200):
            if data[index]['label'] == 0:
                if currplus > 0:
                    currplus -= 1
                    continue
                else:
                    thislen = len(data[index]['content'])
                    transed = ""  # 用于存放翻译后的新闻
                    # if thislen > 5000:  # 谷歌翻译API长度限制，长度>5000的需要分段翻译 但考虑到\n，可对所有新闻执行类似操作
                    print("index{},标签为0，需反向翻译，新闻长度为{}：".format(index, thislen))
                    sp = data[index]['content'].split('\n')  # 按照段落分段
                    maxsp = list()  # 按照不大于5000的长度存放的分段
                    for spp in sp:
                        if len(spp) > 5000:
                            print("分段长度大于5000！")  # googletrans限制5000字符
                            print(spp)
                            spoint = spp.split('.')  # 按照句点分句
                            for sppp in spoint:
                                if len(sppp) > 5000:
                                    print("分句长度大于5000！")  # googletrans限制5000字符
                                    exit()
                                else:
                                    if size(maxsp) == 0:
                                        if len(spp) != 0:
                                            maxsp.append(sppp)
                                    elif len(sppp) + len(maxsp[-1]) < 5000:
                                        if len(spp) != 0:
                                            maxsp[-1] = maxsp[-1] + sppp
                                    else:
                                        maxsp.append(sppp)
                            continue  # 大于5000特殊处理
                        if size(maxsp) == 0:  # 开头直接加入
                            maxsp.append(spp)
                        else:
                            if len(spp) + len(maxsp[-1]) < 5000:  # 组合成不超过5000的长度的大分段
                                if len(spp) != 0:
                                    maxsp[-1] = maxsp[-1] + spp
                            else:
                                maxsp.append(spp)
                    for spp in maxsp:
                        print("分段长度{}，内容：{}".format(len(spp), spp))
                        tmp = googletranslator.translate(spp, dest='zh-cn').text
                        transed = transed + googletranslator.translate(tmp, dest='en').text
                    newitem = {}
                    newitem['content'] = transed
                    newitem['label'] = 0
                    data.append(newitem)
                    print("反向翻译后新闻：{}".format(transed))
                    out = json.dumps(data, ensure_ascii=False)
                    f2 = open("data\\"+filename, 'w', encoding='UTF-8')
                    f2.write(out)
                    f2.close()
                    print("输出文件已更新！")

        # 输出翻译后的为json文件
        out = json.dumps(data, ensure_ascii=False)
        f2 = open("data\\"+filename, 'w', encoding='UTF-8')
        f2.write(out)
        f2.close()
        print("反向翻译完成！")



