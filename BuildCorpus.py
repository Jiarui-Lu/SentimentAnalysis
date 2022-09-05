"""
第一步:
构建语料库和情感词典
采取Word2vec
生成词向量矩阵
输出result文件库中的Word2vec_model

"""
from pathlib import Path
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
import multiprocessing
import numpy as np

neg = []
pos = []
cpu_count = multiprocessing.cpu_count()
vocab_dim = 100
n_exposures = 5 # 所有频数超过10的词语
window_size = 7
# n_epoch = 4
# input_length = 100
maxlen = 100

def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries

    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        #  freqxiao10->0 所以k+1
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引,(k->v)=>(v->k)
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量, (word->model(word))

        def parse_dataset(combined): # 闭包-->临时使用
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0) # freqxiao10->0
                data.append(new_txt)
            return data # word=>index
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec,combined
    else:
        print('No data provided...')

# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):

    model = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count)
    model.build_vocab(combined)  # input: list
    model.train(combined,total_examples=model.corpus_count,epochs=50)
    model.save(r'result\Word2vec_model.pkl')
    index_dict, word_vectors, combined = create_dictionaries(model=model, combined=combined)
    return index_dict, word_vectors, combined

def extract(f, result):
    for line in f:
        l = list(line.strip())
        for i in l:
            result.append(i)

def SelectChinese(result):
    new_result = []
    for word in result:
        if '\u4e00' <= word <= '\u9fff':
            new_result.append(word)
    return new_result


# # 提取原始消极和积极词典的单个字
with Path(r'data\tsinghua.negative.txt').open() as f:
    extract(f, neg)
with Path(r'data\tsinghua.positive.txt').open() as g:
    extract(g, pos)

new_pos=SelectChinese(pos)
new_neg=SelectChinese(neg)
combined = np.concatenate((new_pos,new_neg))
# print(combined)
y = np.concatenate((np.ones(len(new_pos), dtype=int), np.zeros(len(new_neg),dtype=int)))
# print(y)
index_dict, word_vectors,combined=word2vec_train(combined)
# print(index_dict)
# print(word_vectors)
# print(combined)

