import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from utils.data_tidy import UserData, SongData

def get_song_list(song):
    # 建立song的dict() & 紀錄song id
    song_list = dict()
    song_id_list = []

    for index, row in song.df_meta_song.iterrows():
        song_id = row['song_id']
        song_id_list.append(song_id)

        # 檢查 song_id 是否存在於 song_list 中
        if song_id in song_list:
            pass
        else:
            # 將相應的資訊填充到 song_list 中
            meta = dict()
            meta['song_id'] = row['song_id']
            meta['artist_id'] = row['artist_id']
            meta['song_length'] = row['song_length']
            meta['album_id'] = row['album_id']
            meta['language_id'] = row['language_id']
            song_list[song_id] = meta

    ########### composer #################
    for index, row in song.df_meta_song_composer.iterrows():
        song_id = row['song_id']

        if song_id in song_list:
            song_list[song_id]['composer_id'] = row['composer_id']
        else:
            song_id_list.append(song_id)
            meta = dict()
            meta['song_id'] = row['song_id']
            meta['artist_id'] = None
            meta['song_length'] = None
            meta['album_id'] = None
            meta['language_id'] = None
            meta['composer_id'] = row['composer_id']
            song_list[song_id] = meta

    ############# genre #############
    for index, row in song.df_meta_song_genre.iterrows():
        song_id = row['song_id']

        if song_id in song_list:
            song_list[song_id]['genre_id'] = row['genre_id']
        else:
            song_id_list.append(song_id)
            id.append(song_id)
            meta = dict()
            meta['song_id'] = row['song_id']
            meta['artist_id'] = None
            meta['song_length'] = None
            meta['album_id'] = None
            meta['language_id'] = None
            meta['composer_id'] = None
            meta['genre_id'] = row['genre_id']
            song_list[song_id] = meta

 

    ######### lyricist ################3
    for index, row in song.df_meta_song_lyricist.iterrows():
        song_id = row['song_id']

        if song_id in song_list:
            song_list[song_id]['lyricist_id'] = row['lyricist_id']
        else:
            song_id_list.append(song_id)
            meta = dict()
            meta['song_id'] = row['song_id']
            meta['artist_id'] = None
            meta['song_length'] = None
            meta['album_id'] = None
            meta['language_id'] = None
            meta['genre_id'] = None
            meta['composer_id'] = None
            meta['lyricist_id'] = row['lyricist_id']
            song_list[song_id] = meta

    #################### producer #############
    for index, row in song.df_meta_song_producer.iterrows():
        song_id = row['song_id']

        if song_id in song_list:
            song_list[song_id]['producer_id'] = row['producer_id']
        else:
            song_id_list.append(song_id)
            meta = dict()
            meta['song_id'] = row['song_id']
            meta['artist_id'] = None
            meta['song_length'] = None
            meta['album_id'] = None
            meta['language_id'] = None
            meta['genre_id'] = None
            meta['composer_id'] = None
            meta['lyricist_id'] = None
            meta['producer_id'] = row['producer_id']
            song_list[song_id] = meta

    ############# titletext ############
    for index, row in song.df_meta_song_titletext.iterrows():
        song_id = row['song_id']

        if song_id in song_list:
            song_list[song_id]['title_text_id'] = row['title_text_id']
        else:
            song_id_list.append(song_id)
            meta = dict()
            meta['song_id'] = row['song_id']
            meta['artist_id'] = None
            meta['song_length'] = None
            meta['album_id'] = None
            meta['language_id'] = None
            meta['genre_id'] = None
            meta['composer_id'] = None
            meta['lyricist_id'] = None
            meta['producer_id'] = None
            meta['title_text_id'] = row['producer_id']
            song_list[song_id] = meta

    df_song_list = pd.DataFrame(list(song_list.values()))
    #print("suceed")
    return df_song_list

def get_merge_data(train_data, song_list_data):
  user_song_df = pd.merge(train_data, song_list_data.drop_duplicates(['song_id']), on='song_id', how='left')
  return user_song_df

def create_listen_count(user_song_df):
    user_song_df['song_unique'] = user_song_df['song_id']
    # 累加歌曲聆聽量
    song_grouped = user_song_df.groupby(['song_unique']).agg({'song_id':'count'}).reset_index()
    song_grouped.rename(columns={'song_id': 'song_id_listen_count'}, inplace=True)

    grouped_sum = song_grouped['song_id_listen_count'].sum()
    # 計算歌曲聆聽量佔train source的百分比
    song_grouped['percentage'] = (song_grouped['song_id_listen_count'] / grouped_sum ) * 100  
    # 照聆聽量排列
    song_grouped.sort_values(['song_id_listen_count', 'song_unique'], ascending=[0,1])

    return song_grouped


def calculate_similarity_matrix(user_song_df, df_song_list):
     # 在原始數據框中添加 'listen_count' 列
    user_song_df['listen_count'] = user_song_df.groupby(['session_id', 'song_id'])['song_id'].transform('count')
    # 刪掉重複的 'session_id' 和 'song_id'
    user_song_df_filtered = user_song_df.drop_duplicates(subset=['session_id', 'song_id', 'listen_count']).reset_index(drop = True)
        

    # 兩個df分別要drop的column    
    user_song_df_filtered.drop(columns = ['unix_played_at', 'play_status', 'login_type', 'listening_order', 
                                          'song_length', 'song_unique', 'producer_id', 'composer_id', 'lyricist_id', 'listen_count'], inplace=True)

    df_song_list.drop(columns = ['song_length', 'producer_id', 'composer_id', 'lyricist_id'], inplace=True)


    # print(user_song_df_filtered.head(15))
    ###### 先減少資料量測試
    df_song_list = df_song_list.drop(labels = range(10000,1030712), axis = 0)
    user_song_df_filtered = user_song_df_filtered.drop(labels = range(10000,9912719), axis = 0)


   # 填補缺失值
    df_song_list['title_text_id'].fillna('', inplace=True)

    # 創建一個 SimpleImputer 物件，使用最常見的值填充缺失值
    imputer = SimpleImputer(strategy='most_frequent')
    df_song_list['artist_id'] = imputer.fit_transform(df_song_list['artist_id'].values.reshape(-1, 1)).ravel()

    imputer = SimpleImputer(strategy='most_frequent')
    df_song_list['language_id'] = imputer.fit_transform(df_song_list['language_id'].values.reshape(-1, 1)).ravel()

    # 使用 TF-IDF 轉換
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_title = tfidf_vectorizer.fit_transform(df_song_list['title_text_id'])

    # 對 artist_id 和 language_id 使用標準化（Standardization）
    scaler = StandardScaler()
    scaled_artist_id = scaler.fit_transform(df_song_list['artist_id'].values.reshape(-1, 1))
    # scaled_language_id = scaler.fit_transform(df_song_list['language_id'].values.reshape(-1, 1))

    # 將 artist_id 和 language_id 的特徵整合到 TF-IDF 特徵中
    # tfidf_matrix_artist_language = hstack([tfidf_matrix_title, scaled_artist_id, scaled_language_id])
    # 將 artist_id 和 language_id 的特徵整合到 TF-IDF 特徵中
    # tfidf_matrix_artist_language = hstack([tfidf_matrix_title, csr_matrix(scaled_artist_id), csr_matrix(scaled_language_id)])
    tfidf_matrix_artist_language = hstack([tfidf_matrix_title, csr_matrix(scaled_artist_id)])


    # 使用 linear_kernel 計算相似性
    cosine_similarities = linear_kernel(tfidf_matrix_artist_language, tfidf_matrix_artist_language)
    
    return cosine_similarities, user_song_df_filtered


def make_recommend(target_session_id, cos_similarity, user_song_df_filtered):
    
    # 初始化一個空的列表，用於存儲所有目標用戶聽過的歌曲的索引
    all_target_indices = []

    # 收集所有目標用戶聽過的歌曲的索引
    target_indices = user_song_df_filtered[user_song_df_filtered['session_id'] == target_session_id].index
    all_target_indices.extend(target_indices)

    # print(all_target_indices)

    # 找到相似的歌曲
    similar_songs = []
    for target_index in all_target_indices:
        similar_songs.extend(list(enumerate(cos_similarity[target_index])))

    # 根據相似性排序歌曲
    similar_songs = sorted(similar_songs, key=lambda x: x[1], reverse=True)

    # 取前五首歌曲作為推薦
    recommended_songs = similar_songs[0:5]

    return recommended_songs