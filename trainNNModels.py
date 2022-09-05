"""
第二步:
输入训练的100维词向量矩阵权重
训练CNN-LSTM神经网络
双分类标签
训练模型得出准确率
保存模型LSTM_model

"""
import BuildCorpus
import pandas as pd
import tensorflow as tf
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
import yaml



def get_data(index_dict, word_vectors):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))  # 初始化 索引为0的词语，词向量全为0
    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    return n_symbols, embedding_weights


vocab_dim = BuildCorpus.vocab_dim
index_dict = BuildCorpus.index_dict
word_vectors = BuildCorpus.word_vectors
combined = BuildCorpus.combined
y = BuildCorpus.y
n_symbols, embedding_weights = get_data(index_dict, word_vectors)



X_train, X_test, y_train, y_test = train_test_split(combined, y, test_size=0.2, random_state=42)
y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=2)



# 定义网络结构
def CNN_LSTM_Model(batch_size, epochs, input_dim, num_classes):
    model = Sequential()
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        weights=[embedding_weights],
                        input_length=input_dim,
                        trainable=False))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=250, kernel_size=3, padding="valid", activation=tf.nn.relu, strides=1))
    model.add(MaxPooling1D())
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(num_classes, activation=tf.nn.softmax))

    model.compile(loss=categorical_crossentropy, optimizer="adam", metrics=["accuracy"])
    # model.summary()

    model.fit(x=X_train, y=y_train_one_hot, batch_size=batch_size, epochs=epochs,
              validation_data=(X_test, y_test_one_hot))

    model.save(r'result\LSTM_model.h5')

    return model



batch_size = 64
input_dim = 100
num_classes = 2
epochs = 10

model = CNN_LSTM_Model(batch_size=batch_size, epochs=epochs, input_dim=input_dim, num_classes=num_classes)

# model=MLP(batch_size=batch_size,epochs=epochs,input_dim=input_dim,num_classes=num_classes)


# 训练得分和准确度
score, acc = model.evaluate(X_test, y_test_one_hot, batch_size=128)

print("#---------------------------------------------------#")
print("预测得分:{}".format(score))
print("预测准确率:{}".format(acc))
print("#---------------------------------------------------#")
print("\n")

# 模型预测
print("#---------------------------------------------------#")
print("测试集的预测结果，对每个类有一个得分/概率，取值大对应的类别")
predictions = model.predict(X_test[:50])

print(predictions)

for i in range(len(y_test[:50])):
    print(np.argmax(predictions[i]), end=' ')
print()
print(y_test[:50])
print("#---------------------------------------------------#")
print("\n")
