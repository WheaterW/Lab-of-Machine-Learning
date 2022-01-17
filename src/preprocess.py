import json
import os
import string

from googletrans import Translator
from langdetect import detect
from numpy import size
import nltk.tokenize as tk
# 三种词干提取器对比 #
import nltk.stem.porter as pt
import nltk.stem.lancaster as lc
import nltk.stem.snowball as sb
import nltk.stem as ns

class multiFilters:
    def multifilters(self, data, filename):
        '''
        对data进行多种langs的过滤
        :param filename: outout filename
        :param data:    input data
        :param langs:
        :return:
        '''
        langs = ['pt', 'af', 'ru', 'de', 'es', 'zh-cn', 'en']
        global stopwords_all
        SWdict = {}
        lemmatizer = ns.WordNetLemmatizer()  # 词形转换器
        pt_stemmer = pt.PorterStemmer()  # 波特词干提取器——有些过分比如his和hi都是hi
        lc_stemmer = lc.LancasterStemmer()  # 兰卡斯词干提取器
        sb_stemmer = sb.SnowballStemmer("english")  # 思诺博词干提取器
        # res = []
        for lang in langs:  # 存储所需语言的停用词列表
            # print(lang)
            with open("./stopwords/" + lang + ".json", 'r', encoding="UTF-8") as f:
                stopwords = json.load(f)
            filtered_stopwords = []
            # 去掉标点符号的stopwords #
            # for word in stopwords:
            #     filtered_stopwords.append(''.join([char for char in word if char not in string.punctuation]))
            # SWdict[lang] = filtered_stopwords

            # 分词后的stopwords #
            for stopword in stopwords:
                SWdict[lang] = SWdict.get(lang, []) + tk.word_tokenize((stopword))
            # print(SWdict[lang])


        remove_char = string.punctuation + string.digits + "—““‘’"
        remove_char_less = string.digits + ",.'\"/\\:%$()-“”‘’—"
        for datanews in data:
            news = datanews['content']
            lang = detect(news)
            print("语言类型{}，过滤前{}".format(lang, news))
            if lang in SWdict:
                # nltk分词 #
                splitnews = tk.word_tokenize(datanews['content'])
                # 手动分词 # ——删除停用词之前只对分隔符进行split()，避免诸如haven't->have n't 问题：dead.
                splitnews = datanews['content'].split()
                # 词形转换lemma #
                splitnews = [lemmatizer.lemmatize(word, pos='v') if lemmatizer.lemmatize(word, pos='v') != word
                             else lemmatizer.lemmatize(word, pos='n') for word in splitnews]
                # news = "".join(word+" " for word in splitnews if word != " ")
                # 去除停用词——word_tokenize #
                splitnews = [word.lower() for word in splitnews if word.lower() not in SWdict[lang]]
                news = "".join(word+" " for word in splitnews)
                # 词干提取stem #
                # splitnews = [sb_stemmer.stem(word) for word in splitnews]
                # news = "".join(word + " " for word in splitnews if word != " ")
                # 去除标点和数字、字母全转换为小写（因为停用词表全为小写）（注意此时按照char处理，空格也是char，所以不需要加入空格） #
                news = "".join([char.lower() for char in news if char not in remove_char])
                # news = "".join([char.lower() for char in news if char not in remove_char_less]) # 不去除!和?

                datanews['content'] = news
                print("过滤后{}".format(news))
            else:
                continue
        # 输出翻译后的为json文件
        out = json.dumps(data, ensure_ascii=False)
        f2 = open(filename, 'w', encoding='UTF-8')
        f2.write(out)
        f2.close()
        print("过滤完毕！")
        return data


