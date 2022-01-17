import json

from googletrans import Translator
from langdetect import detect
from numpy import size
from nltk import tokenize

class transfile:
    def trans_to_en(self, filename):
        googletranslator = Translator(service_urls=['translate.google.cn', 'translate.google.hk'])
        with open("data/"+filename, 'r', encoding="UTF-8") as f:
            data = json.load(f)
        f.close()
        type, suffix = filename.split("_", 1)
        print(type)
        index = 0

        for td in data:
            if detect(td['content']) != 'en':  # 语言检测，非英语需要翻译
                if index == 81 and type == 'test':  # 测试集81号是xml代码
                    index += 1
                    continue
                thislen = len(td['content'])
                transed = ""  # 用于存放翻译后的新闻
                # if thislen > 5000:  # 谷歌翻译API长度限制，长度>5000的需要分段翻译 但考虑到\n，可对所有新闻执行类似操作
                print("index{},非英语新闻，需翻译，新闻长度为{}：".format(index, thislen))
                sp = td['content'].split('\n')  # 按照段落分段
                maxsp = list()
                for spp in sp:
                    if len(spp) > 5000:
                        print("分段长度大于5000！")  # googletrans限制5000字符
                        print(spp)
                        spoint = tokenize.sent_tokenize(spp)
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
                    transed = transed + googletranslator.translate(spp, dest='en').text
                td['content'] = transed
                print("翻译后新闻：{}".format(transed))
                out = json.dumps(data, ensure_ascii=False)
                f2 = open("data/"+filename, 'w', encoding='UTF-8')
                f2.write(out)
                f2.close()
                print("输出文件已更新！")
            index += 1

        # 输出翻译后的为json文件
        out = json.dumps(data, ensure_ascii=False)
        f2 = open("data/"+filename, 'w', encoding='UTF-8')
        f2.write(out)
        f2.close()
        print("翻译完成！")
