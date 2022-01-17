import numpy as np

class Transformer:
    def data(self, rawdata):
        res = []
        for item in rawdata:
            res.append(item['content'])
        return np.array(res)

    def label(self, rawdata):
        res = []
        for item in rawdata:
            res.append(item['label'])
        return np.array(res)
