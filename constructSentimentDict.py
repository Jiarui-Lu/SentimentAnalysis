"""
第四步:
导入stocknews新闻文本数据
对每天的数据进行groupby分类
形成原始文本字典
调取训练好的模型对每天的文本进行情感分析
结果保存至sentiment_data

"""
import os
import datetime
import pandas as pd
import time
import Classification

def SelectChinese(result):
    new_result = ''
    for word in result:
        if '\u4e00' <= word <= '\u9fff':
            new_result+=word
    return new_result

BASE_DIR=r'data\stockNews'

# 获取新闻的所有时间
# 获取time中每一天的所有公司新闻的内容
stock_dirs=os.listdir(BASE_DIR)
content=dict()
for stock in stock_dirs:
    stock_path=os.path.join(BASE_DIR,stock)
    news_dirs=os.listdir(stock_path)
    for news in news_dirs:
        news_time=news.strip('.txt')
        content[news_time]=[]
        news_path=os.path.join(stock_path,news)
        with open(news_path,encoding='utf-8') as f:
            for line in f:
                content[news_time].append(line)
    print('{} is done'.format(stock))

# content={'2021-01-01':['我和你','心连心'],'2021-01-02':['共住地球村','我。吃。。饭。了。','【ssssss】']}
new_content=dict()
# 处理原始字典
for k,v in content.items():
    s=''.join(v)
    s=SelectChinese(s)
    new_content[k]=s
new_content=dict(sorted(new_content.items(), key=lambda x:x[0]))




data=pd.read_excel(r'data\HS300data.xls',index_col=1)
data['time_str']=[str(t)[:10] for t in data.index]


sentiment_dict=dict()
for t in data['time_str'][300:]:
    sentiment=Classification.lstm_predict(new_content[t])
    sentiment_dict[t]=sentiment
#
sentiment_data=pd.DataFrame(sentiment_dict.values(),index=sentiment_dict.keys(),columns=['sentiment'])
sentiment_data.to_excel(r'result\sentiment_data_2.xls')






















