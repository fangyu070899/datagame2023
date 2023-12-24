from collections import Counter
import pandas as pd
import numpy as np
import json
import math
import random

from utils.data_analyzing import Data
from utils.special_cases import Case
from utils.similarity import Similarity

class Predict:
    def __init__(self) -> None:
        self.data = Data()
        self.case = Case(self.data.test_source)

    def predict(self):
        self.data.reset_target_json()
        self.data.write_result()

        # 直接預測相似的
        for key, value in self.data.similarity.items():
            id = value['max_key']
            new_values = self.data.training_target[id]
            self.data.test_target[key] = new_values
            # self.data.record_selected_song(new_values)
            print(f'sim : {key}')

        # 預測 special cases
        for key in self.data.test_source.keys():
            if key not in self.data.similarity.keys(): 
                print(f'special cases : {key}')
                result = self.predict_songs(key)
                self.data.test_target[key] = result
                # self.data.record_selected_song(result)

        for key, value in self.data.test_target.items():
            # 塞 cb
            print(f'cb : {key}')
            num = 0
            for i in range(len(value)):
                if self.data.test_target[key][i]  == None:
                    self.data.test_target[key][i] = self.data.cb[key][num]
                    # self.data.record_selected_song(list(self.data.cb[key][num]))
                    num+=1

        # 塞 last5 與 random
        for key, value in self.data.test_target.items():
            # 檢查有沒有重複
            # 有就拿掉塞 last5  與 random
            result = []
            num = 19
            for i in range(len(value)):
                if self.data.test_target[key][i]  not in result:
                    result.append(self.data.test_target[key][i])
            
            if len(result) < 5:
                for i in range(5):
                    if len(result)>=5: break
                    num = 19-i
                    if self.data.test_source[key][num] not in result:
                        result.extend(self.data.test_source[key][num])
                        # self.data.record_selected_song(list(self.data.test_source[key][num]))
                if len(result)<5:
                    # result.extend(self.pick_random_differently(5-len(result)))
                    result.extend(self.pick_random(5-len(result)))
            
            if len(result) > 5:
                result = result[0:5]
            
            self.data.test_target[key] = result
            
        self.data.write_result()

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

    def pick_random_differently(self, num):
        song_data = self.data.df_meta_song['song_id']
        selected_songs = self.data.selected_songs.keys()

        if (len(song_data) - len(selected_songs)) < num:
            return self.pick_random(5)
        songs = song_data[~song_data.isin(selected_songs)]

        random = songs.sample(num).values
        self.data.record_selected_song(random)
        
        print(f'random : {random}')
        return random