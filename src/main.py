# Coded By Wei Yunze , 未启用的模块处于注释状态#
# 通用 #
import json
import random
import string
import nltk
import numpy as np
import pandas
from numpy import size
# 翻译相关 #
from langdetect import detect, detect_langs
from translator import transfile    # 自定义包
# 预处理相关 #
from datastrengthen import Datastengthen    # 自定义包
from json2ndarray import Transformer    # 自定义包
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from preprocess import multiFilters # 自定义包
# 模型相关 #
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
# 模型评估相关 #
from sklearn.model_selection import cross_val_score

if __name__ == '__main__':

# 训练集分析 #

    # langs = {}
    # labels = {}
    # dataframe = pandas.read_json("data\\train_translated_plus400.json")
    # for i in range (0, size(dataframe['content'])):
    #     langs[detect(dataframe['content'][i])] = langs.get(detect(dataframe['content'][i]), 0) + 1
    #     labels[str(dataframe['label'][i])] = labels.get(str(dataframe['label'][i]), 0) + 1
    # print(langs)
    # print(labels)
    # exit()

# 基于反向翻译的数据增强 #

    # strength = Datastengthen()
    # strength.stengthbytranspart("train_translated_plus800.json", 0.2)
    # strength.stengthbytranswhole("train_translated_plus200_all.json")
    # exit()

# 翻译相关处理 #

    # transworker = transfile()
    # print("开始翻译训练集：")
    # transworker.trans_to_en("train_newtrans.json") # 翻译到原文件，实现断点续传
    # print("开始翻译测试集：")
    # transworker.trans_to_en("test_newtrans.json")  # 翻译到原文件，实现断点续传
    # exit()


# 训练数据集读取 #

    # with open("data/train.json", 'r', encoding="UTF-8") as f:
    # with open("data/train_translated.json", 'r', encoding="UTF-8") as f:
    # with open("data/train_translated_plus400.json", 'r', encoding="UTF-8") as f:
    # with open("data/train_translated_plus400_sw.json", 'r', encoding="UTF-8") as f:
    # with open("data/train_translated_plus400_sw_pd.json", 'r', encoding="UTF-8") as f:
    # with open("data/train_translated_plus400_pd.json", 'r', encoding="UTF-8") as f:
    # with open("data/train_translated_plus400_lemma_sw.json", 'r', encoding="UTF-8") as f:
    with open("data/train_translated_plus400_lemma_pd_sw.json", 'r', encoding="UTF-8") as f:
    # with open("data/train_translated_plus400_stem_pd_sw.json", 'r', encoding="UTF-8") as f:
        train_data = json.load(f)
    f.close()

# 预测数据集读取 #

    # with open("data/test.json", 'r', encoding="UTF-8") as f:
    # with open("data/test_translated.json", 'r', encoding="UTF-8") as f:
    # with open("data/test_translated_sw.json", 'r', encoding="UTF-8") as f:
    # with open("data/test_translated_sw_pd.json", 'r', encoding="UTF-8") as f:
    # with open("data/test_translated_pd.json", 'r', encoding="UTF-8") as f:
    # with open("data/test_translated_lemma_sw.json", 'r', encoding="UTF-8") as f:
    with open("data/test_translated_lemma_pd_sw.json", 'r', encoding="UTF-8") as f:
    # with open("data/test_translated_stem_pd_sw.json", 'r', encoding="UTF-8") as f:
        test_data = json.load(f)
    f.close()


# 过滤器——过滤数据集并存储为文件 #

    # Preprocessor = multiFilters()
    # train_data = Preprocessor.multifilters(train_data, "data/train_translated_plus400_stem_pd_sw.json")
    # train_data = Preprocessor.multifilters(train_data, "data/test_translated_filtered_stem+rc.json")
    # test_data = Preprocessor.multifilters(test_data, "data/test_translated_stem_pd_sw.json")
    # test_data = Preprocessor.multifilters(test_data, "data/test_translated_filtered_stem+rc.json")
    # exit()

# 将json转换为ndarray #

    transformer = Transformer()
    train_label = transformer.label(train_data)
    train_data = transformer.data(train_data)
    test_data = transformer.data(test_data)

# Pipeline of KNN, GridSearch #

    # pl_knn = Pipeline([('cv', CountVectorizer(max_features=10000)),
    #                    ('ss', StandardScaler(with_mean=False)),
    #                    ('tf', TfidfTransformer()),
    #                    ('knn', KNeighborsClassifier())])
    # para_grid = {
    #     'cv__max_features': np.arange(4000, 8001, 2000),
    #     'knn__n_neighbors': np.arange(3, 14, 2)
    # }
    # grid_res = GridSearchCV(pl_knn, para_grid, scoring='f1_macro', cv=3)
    # grid_res.fit(train_data, train_label)
    # print(grid_res.best_params_)
    # print(grid_res.best_score_)
    # res = grid_res.predict(test_data)
    # exit()

# Pipeline of KNN, Classifier #

    # pl_knn = Pipeline([('cv', CountVectorizer(max_features=8000)),
    #                    ('ss', StandardScaler(with_mean=False)),
    #                    ('tf', TfidfTransformer()),
    #                    ('knn', KNeighborsClassifier(n_neighbors=3))])
    # score = cross_val_score(pl_knn, train_data, train_label, scoring='f1_macro', cv=3)
    # print(score.mean())
    # pl_knn.fit(train_data, train_label)
    # res = pl_knn.predict(test_data)

# Pipeline of NB, GridSearch #

    # pl_nb = Pipeline([('cv', CountVectorizer(max_features=10000)),
    #                    ('tf', TfidfTransformer()),
    #                    ('nb', MultinomialNB())])
    # para_grid = {
    #     # 'cv__max_features': np.arange(3000, 8001, 1000),
    #     'nb__alpha': np.arange(0.001, 0.05, 0.002)
    # }
    # grid_res = GridSearchCV(pl_nb, para_grid, scoring='f1_macro', cv=3)
    # grid_res.fit(train_data, train_label)
    # print(grid_res.best_params_)
    # print(grid_res.best_score_)
    # res = grid_res.predict(test_data)
    # exit()

# Pipeline of NB, Classifier #

    # pl_nb = Pipeline([('cv', CountVectorizer(max_features=10000)),
    #                    ('tf', TfidfTransformer()),
    #                    ('nb', BernoulliNB(alpha=0.04))])
    # score = cross_val_score(pl_nb, train_data, train_label, scoring='f1_macro', cv=5)
    # print(score)
    # pl_nb.fit(train_data, train_label)
    # res = pl_nb.predict(test_data)

# Pipeline of SVC, GridSearch #

    # pl_svc = Pipeline([('cv', CountVectorizer()),
    #                    ('tf', TfidfTransformer()),
    #                    ('svc', SVC(class_weight='balanced', max_iter=1000))])
    # para_grid = {
    #     'cv__max_features': np.arange(3000, 8001, 3000),
    #     'svc__gamma': [1, 0.1, 0.01],
    #     'svc__C': [1, 10, 100],
    #     'svc__max_iter': [500],
    #     'svc__kernel': ['rbf']
    # }
    # grid_res = GridSearchCV(pl_svc, para_grid, scoring='f1_macro', cv=5)
    # grid_res.fit(train_data, train_label)
    # print(grid_res.best_params_)
    # print(grid_res.best_score_)
    # res = grid_res.predict(test_data)
    # exit()

# Pipeline of SVC, Classifier #

    # pl_svc = Pipeline([('cv', CountVectorizer(max_features=3000)),
    #                    ('tf', TfidfTransformer()),
    #                    ('svc', SVC(class_weight='balanced', max_iter=1000, C=10, gamma=0.1, kernel='rbf'))])
    # pl_svc.fit(train_data, train_label)
    # score = cross_val_score(pl_svc, train_data, train_label, scoring='f1_macro', cv=5)
    # print(score.mean())
    # res = pl_svc.predict(test_data)

# Pipeline of LogisticRegression, GridSearch #

    # pl_lr = Pipeline([('cv', CountVectorizer()),
    #                   ('tf', TfidfTransformer()),
    #                   ('clf', LogisticRegression(class_weight='balanced', random_state=0, max_iter=1000))])
    # para_grid = {
    #     'cv__max_features': np.arange(5000, 12000, 2000),
    #     'clf__C': np.arange(0.7, 2.1, 0.2),
    # }
    # grid_res = GridSearchCV(pl_lr, para_grid, scoring='f1_macro', cv=5)
    # grid_res.fit(train_data, train_label)
    # print(grid_res.best_params_)
    # print(grid_res.best_score_)
    # res = grid_res.predict(test_data)

# Pipline of LogisticRegression, Classifier #

    pl_lr = Pipeline([('cv', CountVectorizer(max_features=7000)),
                      ('tf', TfidfTransformer()),
                      ('clf', LogisticRegression(C=1.3, class_weight='balanced', random_state=0, max_iter=1000))])
    score = cross_val_score(pl_lr, train_data, train_label, scoring='f1_macro', cv=5)
    print(score.mean())
    pl_lr.fit(train_data, train_label)
    res = pl_lr.predict(test_data)

# 输出预测结果到文件 #

    f = open('res.txt', 'w')
    for i in res:
        f.write(str(i) + '\n')
    f.close()

    print("预测结果已输出到文件")

# 先前的测试 #
# 测试用 #
import re
from nltk.corpus import stopwords
import nltk.tokenize as tk
# 三种词干提取器对比 #
import nltk.stem.porter as pt
import nltk.stem.lancaster as lc
import nltk.stem.snowball as sb
# 词型还原 #
import nltk.stem as ns
# 同义词替换 #
from nltk.corpus import wordnet
from youdaoAPI import YoudaoAPI # 付费有道翻译API #
# 词形转换和词干提取的测试 #

    # pt_stemmer = pt.PorterStemmer()  # 波特词干提取器——有些过分比如his和hi都是hi
    # lc_stemmer = lc.LancasterStemmer()  # 兰卡斯词干提取器
    # sb_stemmer = sb.SnowballStemmer("english")  # 思诺博词干提取器
    # lemmatizer = ns.WordNetLemmatizer() # 词形转换器
    # # 不标注词性，哪个不一样就用哪个 #
    # print(lemmatizer.lemmatize("dad's", pos='v')) # meatures，states
    # print(lemmatizer.lemmatize("dad's", pos='n')) # meatures，states
    # print(sb_stemmer.stem("dad's"))
    # print(sb_stemmer.stem("dad's"))
    # print('HQZL'.lower())
    # test = " 123, hqzl  123 "
    # print(test.split(' '))
    # print(tk.word_tokenize(test))
    # tokenizer = tk.WordPunctTokenizer()
    # print(tokenizer.tokenize(test))
    # exit()

# Old #
# 模型优化器——调参 #
    # Optimizor = optimizor()

    # Optimizor.findeta_iter(train_data, train_label)
    # exit()

    # besteta0, bestiter = findeta_iter(train_data, train_label)

    # findalpha(train_data, train_label)
    # exit()

    # Optimizor.GridSearchSCV(train_data=train_data, train_label=train_label)
    # exit()

    # classifierBNB = Bernoulli()
    # classifierGNB = Gaussian()
    # classifierMNB = Multinomial()
    # classifierKNN = Knn()
    # classifierPC = PC()
    # classifierSVC = LinearSVM()

    # res = classifierBNB.classifier(train_data, train_label, test_data, 0.035)
    # res = classifierGNB.classifier(train_data, train_label, test_data)
    # res = classifierMNB.classifier(train_data, train_label, test_data, 0.0117)
    # res = classifierKNN.classifier(train_data, train_label, test_data)
    # res = classifierPC.classifier(train_data, train_label, test_data, besteta0, bestiter)
    # res = classifierPC.classifier(train_data, train_label, test_data, 0.01, 50)
    # res = classifierSVC.classifier(train_data, train_label, test_data)

    # print(roc_auc_score(y_test, y_pred))