* 代码文件包括主文件main.py和五个自定义包文件datastrengthen.py、json2ndarray.py、preprocess.py、translator.py、youdaoAPI.py
  * main.py 中涵盖了实验中涉及的所有功能，**其中处于注释状态的是当前未启用的代码**
  * json2ndarray.py 实现了本数据集json到nddarray的转换
  * youdaoAPI.py 实现了有道翻译API的调用，用于数据增强
  * datastrengthen.py 实现了基于反向翻译的数据增强
  * preprocess.py中实现了多种文本处理方法，**其中处于注释状态的是当前未启用的代码**
  * translator.py实现了基于谷歌翻译的数据集翻译
* /stopwords 文件夹是从网络获取的停用词包
* /data 文件夹是训练、测试用数据，由于训练、测试集体积较大，提交代码包中只包含最优结果对应的处理后数据集