import pandas as pd
import json

"""
1. 整個 session 都是同一首歌的狀況
    target 也是重複這一首歌五次機率
    target 中包含這一首歌機率
2. 整個 session 中有歌曲重複
    （連續）
    target 中包含該歌曲機率
    （不連續）
    target 中包含該歌曲機率
3. 整個 session 有規律 如ABCDEFABCDEF
    target 按照規律 ABCDE
    target 中包含這些首歌 FEBCA
4. 有沒有錯誤欄位
5. 統計meta資訊對session的影響程度
6. 建立歌曲榜單 統計出個個meta資訊中的排名
"""

class Data:
    def __init__(self) :
        self.df_label_train_source = pd.read_parquet(f'data/label_train_source.parquet')
        self.df_label_train_target = pd.read_parquet(f'data/label_train_target.parquet')

        self.df_meta_song = pd.read_parquet(f'data/meta_song.parquet')
        self.df_meta_song_composer = pd.read_parquet(f'data/meta_song_composer.parquet')
        self.df_meta_song_genre = pd.read_parquet(f'data/meta_song_genre.parquet')
        self.df_meta_song_lyricist = pd.read_parquet(f'data/meta_song_lyricist.parquet')
        self.df_meta_song_producer = pd.read_parquet(f'data/meta_song_producer.parquet')
        self.df_meta_song_titletext = pd.read_parquet(f'data/meta_song_titletext.parquet')

    def traverse_training_data(self):
        num_session=0
        for key in self.df_label_train_source.keys():
            songs_list = self.df_label_train_source[key]

            num_session+=1

        print("sessions number: "+ num_session)


    def training_data_to_json(self):
        training_data = {}

        for index, row in self.df_label_train_source.iterrows():
            session_id = row['session_id']
            song_id = row['song_id']
            if session_id not in training_data:
                training_data[session_id] = []  
            
            training_data[session_id].append(song_id)
            print(f"session id: {session_id}   song id: {song_id}")

        with open('data/training_data.json','w') as json_file:
            json.dump(training_data, json_file, indent=2)

    def training_target_to_json(self):
        training_target = {}

        for index, row in self.df_label_train_target.iterrows():
            session_id = row['session_id']
            song_id = row['song_id']
            if session_id not in training_target:
                training_target[session_id] = []  
            
            training_target[session_id].append(song_id)
            print(f"session id: {session_id}   song id: {song_id}")

        with open('data/training_target.json','w') as json_file:
            json.dump(training_target, json_file, indent=2)

