"""
第三步:
主要为三个函数
主要函数为lstm_predict
目的为识别某个长句字符串的情绪
返回对象为布尔值
1为积极
0为消极

"""
import jieba
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
from keras.models import load_model


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
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#
        w2vec = {word: model[word] for word in w2indx.keys()}

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
                        new_txt.append(0)
                data.append(new_txt)
            return data # word=>index
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=100)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec,combined
    else:
        print('No data provided...')


def input_transform(string):
    words=jieba.lcut(string)
    words=np.array(words).reshape(1,-1)
    model=Word2Vec.load(r'result\Word2vec_model.pkl')
    create_dictionaries(model, words)
    _,_,combined=create_dictionaries(model,words)
    return combined

def lstm_predict(string):
    print('loading model......')
    model=load_model(r'result\LSTM_model.h5')

    data=input_transform(string)
    data.reshape(1,-1)
    predictions = model.predict(data)
    # print(predictions) # [[1]]
    sentiment=np.argmax(predictions[0])
    # if sentiment==1:
    #     print(string,' positive')
    # else:
    #     print(string,' negative')
    return sentiment

# lstm_predict('我是那种小鸟胃的淑女类型')


