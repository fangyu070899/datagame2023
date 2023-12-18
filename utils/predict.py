from collections import Counter
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import json
import math
import random

from utils.data_analyzing import Data
from utils.special_cases import Case

class Predict:
    def __init__(self) -> None:
        self.data = Data()
        self.case = Case(self.data.test_source)

    def predict(self):
        # df = pd.DataFrame()
        df = pd.read_csv('special_cases.csv')

        # # count=0
        # for key in self.data.test_source.keys():
        #     # if count>10: break
        #     print(key)
        #     result = self.predict_songs(key)
        #     print(result)
        #     result_data = {'session_id': key, 'top1': result[0], 'top2': result[1], 'top3': result[2], 'top4': result[3], 'top5': result[4]}
        #     df = pd.concat([df,pd.DataFrame([result_data])],ignore_index=True)
        #     # count+=1

        # 將已經選擇過的歌記錄下來
        selected_columns = df.iloc[:, 1:6].values.flatten()
        selected_songs=list(set(selected_columns))
        self.data.record_selected_song(selected_songs)

        # print(self.data.df_selected_songs)


        
        # 空格用 random 填補
        for column in df.columns:
            mask = pd.isna(df[column])
            num_missing = mask.sum()
            if num_missing > 0:
                df.loc[mask, column] = self.pick_random_differently(num_missing)



        df.to_csv('output_file.csv', index=False, header=True, columns=['session_id','top1','top2','top3','top4','top5'])

    def predict_songs(self, session_id):
        result = []
        allsame = self.case.check_allsame(session_id)
        if allsame != False:
            result.append(allsame)
        else: 
            repeat = self.case.check_repeat(session_id)
            if repeat != False:
                end = self.case.check_repeat_at_end(session_id)
                if end != False:
                    result.append(end)
                for i in range(len(repeat)):
                    result.append(repeat[i])
                    if len(result) == 5: break

        if len(result) < 5:
            result.extend([None]*(5-len(result)))
        if len(result) > 5:
            result = result[0:5]
        return result

    def pick_random(self, num):
        songs = self.data.df_meta_song['song_id']
        random = songs.sample(num).values
        print(random)
        return random

    # input data class
    def pick_random_differently(self, num):
        song_data = self.data.df_meta_song['song_id']
        selected_songs = self.data.df_selected_songs['song_id']
        # print(selected_songs)

        if (len(song_data) - len(selected_songs)) < num:
            self.data.df_selected_songs = pd.DataFrame(columns=['song_id'])
            selected_songs = self.data.df_selected_songs['song_id']
            print('all songs had been selected.')

        songs_to_exclude = selected_songs.tolist()
        songs = song_data[~song_data.isin(songs_to_exclude)]

        random = songs.sample(num).values
        print(random)

        self.data.record_selected_song(random)
        
        return random