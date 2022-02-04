import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm
from net import myModle
import torch

def softmax( f ):
    f -= np.max(f)
    return np.exp(f) / np.sum(np.exp(f))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Model = torch.load("拉康快乐机.pth",map_location=device)
Model.eval()
wv = Word2Vec.load("words.model").wv
l = len(wv)

def eval():
    words = list(input("请输入初始文本："))
    need_del = []
    for i in words:
        if not (i in wv.index_to_key):
            need_del.append(i)
    for i in need_del:
        words.remove(i)

    data = np.array([])
    for i in words:
        data = np.append(data, wv.key_to_index[i])
    data = np.stack((data,))
    count = int(input("请输入生成词数："))
    for i in tqdm(range(count)):
        x = torch.Tensor(data).to(device)
        y = Model(x)[0][-1]
        y = y.to("cpu")
        p = y.detach().numpy().reshape((-1,))
        p = softmax(p)
        word_ind = np.random.choice(np.arange(l),p=p)
        new_word = wv.index_to_key[word_ind]
        words.append(new_word)
        data = np.append(data,word_ind)
        data = np.stack((data,))

    s = ""
    for i in words:
        s = s + i
    print(s)
