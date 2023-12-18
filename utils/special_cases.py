from collections import Counter
import pandas as pd
import json


"""
1. 整個 session 都是同一首歌的狀況
2. 整個 session 中有歌曲重複
3. 整個 session 為一序列循環 如ABCDEFABCDEF
"""

class Case:
    def __init__(self,data) :
        self.data = data

    def check_allsame(self, session_id):
        songs_list = self.data[session_id]
        cnt = Counter(songs_list)

        # 如果 session 中全部的歌都是同一首
        if cnt.most_common(1)[0][1] == sum(cnt.values()):
            return songs_list[0]
        return False
        
    def check_repeat(self, session_id):
        songs_list = self.data[session_id]
        cnt = Counter(songs_list)

        # 如果 session 中有歌曲重複
        if cnt.most_common(1)[0][1] > 1 :
            most_items = cnt.most_common(5)
            tmp = []
            for i in range(0,len(most_items)):
                if most_items[i][1] > 1:
                    tmp.append(most_items[i][0])
            return tmp
        return False

    def check_repeat_at_end(self,session_id):
        songs_list = self.data[session_id]
        cnt = Counter(songs_list)
        count = 0
        # 最尾端重複
        for i in range(len(songs_list)-1,-1,-1):
            if songs_list[i] == songs_list[-1]:
                count += 1
            else:
                break

        if count > 1:
            return songs_list[-1]
        return False
