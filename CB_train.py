import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from collections import Counter
import pickle
import random
import csv

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


def calculate_song_feature_matrix(df_song_list):
    df_song_list.drop(columns=['song_length', 'producer_id', 'lyricist_id'], inplace=True)

    # df_song_list['title_text_id'].fillna('', inplace=True)
    df_song_list['genre_id'].fillna('', inplace=True)

    # imputer = SimpleImputer(strategy='most_frequent')
    # 建立 SimpleImputer 實例，將 strategy 設為 'constant'，fill_value 設為 0
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    df_song_list['album'] = imputer.fit_transform(df_song_list['album_id'].values.reshape(-1, 1)).ravel()
    df_song_list['artist'] = imputer.fit_transform(df_song_list['artist_id'].values.reshape(-1, 1)).ravel()
    df_song_list['genre'] = df_song_list['genre_id'].astype('category').cat.codes
    df_song_list['song'] = df_song_list['song_id'].astype('category').cat.codes



    ## new add
    # df_song_list['composer'] = df_song_list['composer_id'].astype('category').cat.codes
    # df_song_list['language'] = imputer.fit_transform(df_song_list['language_id'].values.reshape(-1, 1)).ravel()
    # df_song_list['title'] = df_song_list['title_text_id'].astype('category').cat.codes



    scaler = StandardScaler()
    scaled_album = scaler.fit_transform(df_song_list['album'].values.reshape(-1, 1))
    scaled_artist_id = scaler.fit_transform(df_song_list['artist'].values.reshape(-1, 1))
    scaled_genre = scaler.fit_transform(df_song_list['genre'].values.reshape(-1, 1))

    # scaled_title = scaler.fit_transform(df_song_list['title'].values.reshape(-1, 1))
    # scaled_composer = scaler.fit_transform(df_song_list['composer'].values.reshape(-1, 1))
    # scaled_language = scaler.fit_transform(df_song_list['language'].values.reshape(-1, 1))


    matrix_tiltle_artist_genre = hstack([csr_matrix(scaled_album), csr_matrix(scaled_artist_id), csr_matrix(scaled_genre)]).astype(np.float64)
    new_df_song_list = pd.DataFrame()
    new_df_song_list['song_id'] = df_song_list['song_id']
    new_df_song_list['song'] = df_song_list['song']
    new_df_song_list['album'] = df_song_list['album_id']
    new_df_song_list['artist'] = df_song_list['artist_id']
    new_df_song_list['genre'] = df_song_list['genre_id']

    return matrix_tiltle_artist_genre, new_df_song_list

def create_last_user_song(user_song_df):

  # user_song_df['listen_count'] = user_song_df.groupby(['session_id', 'song_id'])['song_id'].transform('count')
  # user_song_df_filtered = user_song_df.drop_duplicates(subset=['session_id', 'song_id', 'listen_count']).reset_index(drop=True)
  user_song_df_filtered = user_song_df.drop(columns=['unix_played_at', 'play_status', 'login_type',
                                        'song_length', 'producer_id', 'composer_id', 'lyricist_id', 'language_id', 'title_text_id'])
  user_song_df_filtered_again = user_song_df_filtered.copy()
  user_song_df_filtered_again = user_song_df_filtered_again[user_song_df_filtered_again['listening_order'] > 10].reset_index()
  # user_song_df_filtered_again = user_song_df_filtered.drop(columns=['artist_id', 'album_id', 'genre_id'])
  user_song_df_filtered_again['song'] = user_song_df_filtered_again['song_id'].astype('category').cat.codes


  return user_song_df_filtered_again

def create_filtered_user_song(user_song_df_filtered_again):
  user_song_df_filtered_again['listen_count'] = user_song_df_filtered_again.groupby(['session_id', 'song_id'])['song_id'].transform('count')
  user_song_df_filtered_final = user_song_df_filtered_again.drop_duplicates(subset=['session_id', 'song_id', 'listen_count']).reset_index(drop=True)
  # user_song_df_filtered.drop(columns=['unix_played_at', 'play_status', 'login_type', 'listening_order',
  #                                       'song_length', 'producer_id', 'composer_id', 'lyricist_id',], inplace=True)
  # user_song_df_filtered_again = user_song_df_filtered.drop(columns=['artist_id', 'album_id', 'language_id',	'genre_id',	'title_text_id'])
  # user_song_df_filtered_again['song'] = user_song_df_filtered_again['song_id'].astype('category').cat.codes

  return user_song_df_filtered_final


def create_user_interaction(user_song_df_filtered_again):
  #user_interactions_list = user_song_filtered_again[['session_id', 'song_id', 'listen_count']].values.tolist()
  #user_interactions_list = user_song_df_filtered_again.groupby('session_id').apply(lambda x: list(zip(x['song_id'], x['listen_count']))).tolist()

  # 創建一個空的字典，用於存儲每個 session_id 的 user_interactions
  user_interactions_dict = {}

  # 遍歷 user_song_df_filtered 中的每一行
  for index, row in user_song_df_filtered_again.iterrows():
      session_id = row['session_id']
      song = row['song']
      listen_count = row['listen_count']

      # 如果 session_id 不在字典中，創建一個新的 entry
      if session_id not in user_interactions_dict:
          user_interactions_dict[session_id] = []

      # 將每首歌的 song_id 和 listen_count 添加到對應的 session_id 中
      user_interactions_dict[session_id].append((song, listen_count))

  return user_interactions_dict



def calculate_user_profile(user_interactions, matrix_artist_language, df_song_list):
    user_profile = np.zeros(matrix_artist_language.shape[1])

    for item_id, rating in user_interactions:
        item_index = df_song_list.index[df_song_list['song'] == item_id][0]
        user_profile += matrix_artist_language[item_index].toarray()[0] * rating

    return user_profile

def count_artist_album(user_song_df_filtered_final, session_id, user_interactions):
    re_song = []
    re_album = []
    re_artist = []

    album_dic = []
    artist_dic = []
    recommended_song_ids = []
    for item_id, rating in user_interactions:
      # item_index = df_song_list.index[df_song_list['song'] == item_id][0]
      if rating >= 3:
        item = user_song_df_filtered_final[user_song_df_filtered_final['song'] == item_id].song_id
        recommended_song_ids.append(item)
        re_song.append(item.iloc[0])
        # count_song +=1

    # print("Count song:recommend", re_song)
    cur_df = user_song_df_filtered_final.copy()
    cur_df = cur_df[cur_df['session_id'] == session_id].reset_index()
    
    # 累加album_id
    counter_album = Counter(cur_df['album_id'])
    # print(counter_album)
    album_df = pd.DataFrame.from_dict(counter_album, orient='index', columns=['count'])
    # print(album_df)
    for index, row in album_df.iterrows():
      count = row['count']
      if count >= 3 and count < 5:
        album_dic.append(index)
        item = df_song_list[df_song_list['album_id'] == index].song_id
        # print("album_len", len(item))
        choose = random.randint(0, len(item))
        re_album.append(item.iloc[choose])
        recommended_song_ids.append(item.iloc[choose])
      if count >= 5:
        for i in range(5):
          album_dic.append(index)
          item = df_song_list[df_song_list['album_id'] == index].song_id
          # print("album_len", len(item))
          choose = random.randint(0, len(item))
          re_album.append(item.iloc[choose])
          recommended_song_ids.append(item.iloc[choose])

    # 累加artist_id
    counter_artist = Counter(cur_df['artist_id'])
    # print(counter_artist)

    artist_df = pd.DataFrame.from_dict(counter_artist, orient='index', columns=['count'])
    # print(artist_df)
    for index, row in artist_df.iterrows():
      count = row['count']
      if count >= 3 and count < 5:
        artist_dic.append(index)
        item = df_song_list[df_song_list['artist_id'] == index].song_id
        # print("artist_len", len(item))
        choose = random.randint(0, len(item))
        re_artist.append(item.iloc[choose])
        recommended_song_ids.append(item.iloc[choose])
      if count >= 5:
        for i in range(5):
          artist_dic.append(index)
          item = df_song_list[df_song_list['artist_id'] == index].song_id
          # print("artist_len", len(item))
          choose = random.randint(0, len(item))
          re_artist.append(item.iloc[choose])
          recommended_song_ids.append(item.iloc[choose])
  

    sim_song = 5 - len(recommended_song_ids)
    # print(sim_song)

    return recommended_song_ids, sim_song


if __name__ == '__main__':

    data = UserData()
    song = SongData()

    session_ids = data.df_label_test_source['session_id'].unique()
    print(session_ids.shape)

    # 把每首歌依照song_id做出song metada的大矩陣
    df_song_list = get_song_list(song)

    # 把user data(session_id)結合song list
    user_song_df = get_merge_data(data.df_label_test_source, df_song_list)

    # df_listen_count = create_listen_count(user_song_df)

    # 所有歌曲的特徵擷取矩陣(title_text_id, artist_id, genre_id)
    song_feature_matrix, new_song_df_list = calculate_song_feature_matrix(df_song_list)
    # with open('/content/drive/MyDrive/DataGame/data/data/song_feature_new.pkl', 'wb') as f:
    #     pickle.dump(song_feature_matrix, f)


    user_song_df_last = create_last_user_song(user_song_df)
    user_song_df_last.head(50)



    # 累積每個session_id中每首歌的聆聽次數
    user_song_df_filtered_final = create_filtered_user_song(user_song_df_last)

    # 改成list
    u_list= create_user_interaction(user_song_df_filtered_final)

    with open('/content/drive/MyDrive/DataGame/result/CB_prediction_try.csv', 'w', newline='') as f:
      column_names = ['session_id', 'top1', 'similarity1', 'top2', 'similarity2', 'top3', 'similarity3', 'top4', 'similarity4', 'top5', 'similarity5']
      writer = csv.DictWriter(f, fieldnames=column_names)

      # 寫入 CSV 標題
      writer.writeheader()

      count = 0
      for session_id in session_ids:
        print(count, ":", session_id)
        user_interactions = u_list[session_id]
        recommended_song_ids, sim_song = count_artist_album(user_song_df_filtered_final, session_id, user_interactions)
       
        # 創建一個 user_profile，透過聚合 item features
        user_profile = calculate_user_profile(user_interactions, song_feature_matrix, df_song_list)
        # 計算 cosine similarity
        similarities = cosine_similarity([user_profile], song_feature_matrix)

        if sim_song > 0:
          recommended_song_indices = np.argsort(similarities[0])[::-1][:sim_song]
          for i in range(sim_song):
            recommended_song_ids.append(df_song_list['song_id'].iloc[recommended_song_indices].iloc[i])
        # recommended_similarities = similarities[0][recommended_song_indices]


        # 寫入 CSV 檔案
        writer.writerow({
            'session_id': session_id,
            'top1': recommended_song_ids[0],  
            'top2': recommended_song_ids[1],
            'top3': recommended_song_ids[2],
            'top4': recommended_song_ids[3],
            'top5': recommended_song_ids[4],
            
        })


        print(count, " Recommended Items of ", session_id, ":")
        for i in range(5): 
            print(f"{i}, songID: {recommended_song_ids[i]}")
            

        print()
        count+=1
