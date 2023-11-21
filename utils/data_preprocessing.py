import pandas as pd
import json

"""
1. 整個 session 都是同一首歌的狀況
    target 也是重複這一首歌五次
    target 中包含這一首歌
2. 
3. 整個 session 有規律
    target 按照規律
    target 中包含這些首歌
4. 有沒有錯誤欄位
5. 統計各個風格、歌手對session的影響程度
"""

df_label_train_source = pd.read_parquet(f'data/label_train_source.parquet')
df_label_train_target = pd.read_parquet(f'data/label_train_target.parquet')

df_meta_song = pd.read_parquet(f'data/meta_song.parquet')
df_meta_song_composer = pd.read_parquet(f'data/meta_song_composer.parquet')
df_meta_song_genre = pd.read_parquet(f'data/meta_song_genre.parquet')
df_meta_song_lyricist = pd.read_parquet(f'data/meta_song_lyricist.parquet')
df_meta_song_producer = pd.read_parquet(f'data/meta_song_producer.parquet')
df_meta_song_titletext = pd.read_parquet(f'data/meta_song_titletext.parquet')

def training_data_to_json():
    training_data = {}

    for index, row in df_label_train_source.iterrows():
        session_id = row['session_id']
        song_id = row['song_id']
        if session_id not in training_data:
            training_data[session_id] = []  
        
        training_data[session_id].append(song_id)
        print(f"session id: {session_id}   song id: {song_id}")

    with open('data/training_data.json','w') as json_file:
        json.dump(training_data, json_file, indent=2)

def training_target_to_json():
    training_target = {}

    for index, row in df_label_train_target.iterrows():
        session_id = row['session_id']
        song_id = row['song_id']
        if session_id not in training_target:
            training_target[session_id] = []  
        
        training_target[session_id].append(song_id)
        print(f"session id: {session_id}   song id: {song_id}")

    with open('data/training_target.json','w') as json_file:
        json.dump(training_target, json_file, indent=2)

