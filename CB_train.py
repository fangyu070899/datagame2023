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
import pickle
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
    df_song_list.drop(columns=['song_length', 'producer_id', 'composer_id', 'lyricist_id'], inplace=True)
    df_song_list['title_text_id'].fillna('', inplace=True)
    df_song_list['genre_id'].fillna('', inplace=True)

    imputer = SimpleImputer(strategy='most_frequent')
    df_song_list['artist_id'] = imputer.fit_transform(df_song_list['artist_id'].values.reshape(-1, 1)).ravel()
    df_song_list['title'] = df_song_list['title_text_id'].astype('category').cat.codes
    df_song_list['genre'] = df_song_list['genre_id'].astype('category').cat.codes
    df_song_list['song'] = df_song_list['song_id'].astype('category').cat.codes

    scaler = StandardScaler()
    scaled_artist_id = scaler.fit_transform(df_song_list['artist_id'].values.reshape(-1, 1))
    scaled_title = scaler.fit_transform(df_song_list['title'].values.reshape(-1, 1))
    scaled_genre = scaler.fit_transform(df_song_list['genre'].values.reshape(-1, 1))

    matrix_tiltle_artist_genre = hstack([csr_matrix(scaled_title), csr_matrix(scaled_artist_id), csr_matrix(scaled_genre)]).astype(np.float32)
  
    return matrix_tiltle_artist_genre

def create_filtered_user_song(user_song_df):
  user_song_df['listen_count'] = user_song_df.groupby(['session_id', 'song_id'])['song_id'].transform('count')
  user_song_df_filtered = user_song_df.drop_duplicates(subset=['session_id', 'song_id', 'listen_count']).reset_index(drop=True)
  user_song_df_filtered.drop(columns=['unix_played_at', 'play_status', 'login_type', 'listening_order',
                                        'song_length', 'producer_id', 'composer_id', 'lyricist_id',], inplace=True)
  user_song_df_filtered_again = user_song_df_filtered.drop(columns=['artist_id', 'album_id', 'language_id',	'genre_id',	'title_text_id'])
  user_song_df_filtered_again['song'] = user_song_df_filtered_again['song_id'].astype('category').cat.codes

  return user_song_df_filtered_again

def create_user_interaction(user_song_df_filtered_again):
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
    song_feature_matrix = calculate_song_feature_matrix(df_song_list)
    with open('/song_feature.pkl', 'wb') as f:
        pickle.dump(song_feature_matrix, f)

    # 累積每個session_id中每首歌的聆聽次數
    user_song_df_filtered = create_filtered_user_song(user_song_df)
    
    # 改成list
    u_list= create_user_interaction(user_song_df_filtered)

    #############        以下為predict       ###########
    with open('/test_target.csv', 'w', newline='') as f:
      column_names = ['session_id', 'top1', 'top2', 'top3', 'top4', 'top5']
      writer = csv.DictWriter(f, fieldnames=column_names)
      
      # 寫入 CSV 標題
      writer.writeheader()

      count = 0
      for session_id in session_ids:
 
        print(count, ":", session_id)
        user_interactions = u_list[session_id]

        # 創建一個 user_profile，透過聚合 item features
        user_profile = calculate_user_profile(user_interactions, song_feature_matrix, df_song_list)

        # 計算 cosine similarity
        similarities = cosine_similarity([user_profile], song_feature_matrix)

        # 取得推薦的 item IDs
        recommended_song_ids = df_song_list['song_id'][np.argsort(similarities[0])[::-1][:5]]

        # 寫入 CSV 檔案
        writer.writerow({
            'session_id': session_id,
            'top1': recommended_song_ids.iloc[0],
            'top2': recommended_song_ids.iloc[1],
            'top3': recommended_song_ids.iloc[2],
            'top4': recommended_song_ids.iloc[3],
            'top5': recommended_song_ids.iloc[4],
        })

        # 顯示推薦
        print("Recommended Items of ", session_id, ":")
        for item_id in recommended_song_ids:
            print(f"Item {item_id}")

        count+=1

