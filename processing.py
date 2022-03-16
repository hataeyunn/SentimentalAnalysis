from konlpy.tag import Mecab
from load import _load
import numpy as np

def processing():
    stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게', '만', '게임', '겜', '되', '음', '면']

    mecab = Mecab(dicpath=r"C:/mecab/mecab-ko-dic")
    train_data,test_data = _load('dataset/ratings_train.txt','dataset/ratings_test.txt','dataset/new_train.txt')
    train_data['tokenized'] = train_data['reviews'].apply(mecab.morphs)
    train_data['tokenized'] = train_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])
    test_data['tokenized'] = test_data['reviews'].apply(mecab.morphs)
    test_data['tokenized'] = test_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])

    negative_words = np.hstack(train_data[train_data.label == 0]['tokenized'].values)
    positive_words = np.hstack(train_data[train_data.label == 1]['tokenized'].values)

    X_train = train_data['tokenized'].values
    y_train = train_data['label'].values
    X_test= test_data['tokenized'].values
    y_test = test_data['label'].values

    return (X_train,y_train,X_test,y_test)
