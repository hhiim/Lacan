import numpy as np
from torch.utils.data import Dataset
from gensim.models import Word2Vec

class TextData(Dataset):
    def __init__(self, data_path, model_path, s_len):
        f = open(data_path,"r",encoding='utf-8')
        text = f.read()
        f.close()
        words = list(text)
        wv = Word2Vec.load(model_path).wv
        self.dict_len = len(wv)

        need_del = []
        for i in words:
            if not (i in wv.index_to_key):
                need_del.append(i)
        for i in need_del:
            words.remove(i)

        self.data = np.array([])
        for i in words:
            self.data = np.append(self.data,wv.key_to_index[i])
        self.len = s_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if index <= len(self.data)-self.len:
            data = self.data[index:index+self.len]
            return data[0:self.len]
        else:
            return self.__getitem__(index -
                                    (len(self.data)-self.len))