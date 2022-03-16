import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def _load(load_1,load_2,load_3):
    total_data_movie_train = pd.read_table(load_1, names=['id', 'reviews','label'])
    total_data_movie_test = pd.read_table(load_2, names=['id', 'reviews','label'])

    total_data_movie_train = total_data_movie_train.drop(['id'],axis=1)
    total_data_movie_test = total_data_movie_test.drop(['id'],axis=1)

    total_data = pd.read_table(load_3, names=['label', 'reviews'])

    total_data = pd.concat([total_data_movie_train,total_data_movie_test,total_data])

    total_data.drop_duplicates(subset=['reviews'], inplace=True)

    train_data, test_data = train_test_split(total_data, test_size = 0.1, random_state = 42)

    train_data['reviews'] = train_data['reviews'].str.replace(r"[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","",regex=True)
    train_data['reviews'].replace('', np.nan, inplace=True)
    train_data = train_data.dropna(how='any') # Null 값 제거

    test_data['reviews'] = test_data['reviews'].str.replace(r"[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","",regex=True)
    test_data['reviews'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
    test_data = test_data.dropna(how='any') # Null 값 제거

    return (train_data,test_data)

