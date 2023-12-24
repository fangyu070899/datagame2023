from collections import Counter
import pandas as pd
import json

"""
1. 整個 session 都是同一首歌的狀況
    target 也是重複這一首歌五次機率
    target 中包含這一首歌機率
2. 整個 session 中有歌曲重複
    （取前五首）
    target 中包含該五首歌曲機率（分 第一名 第二名...）
    （在 session 末尾重複）
    target 中包含該歌曲機率
3. 整個 session 為一序列循環 如ABCDEFABCDEF
    target 按照規律接續 ABCDE
    target 中包含這些首歌 FEBCA
4. 整個 session 有重複子序列 如ABCABCD
    target 中包含此序列 ABC
    target 中包含這些首歌 
5. 有沒有錯誤欄位
6. 統計meta資訊對session的影響程度
7. 建立歌曲榜單 統計出個個meta資訊中的排名
8. 統計 target 的重複狀況 
"""

class Data:
    def __init__(self) :
        # self.df_label_train_source = pd.read_parquet(f'data/label_train_source.parquet')
        # self.df_label_train_target = pd.read_parquet(f'data/label_train_target.parquet')
        # self.df_label_test_source = pd.read_parquet(f'data/label_test_source.parquet')

        self.df_meta_song = pd.read_parquet(f'data/meta_song.parquet')
        
        # self.df_meta_song_composer = pd.read_parquet(f'data/meta_song_composer.parquet')
        # self.df_meta_song_genre = pd.read_parquet(f'data/meta_song_genre.parquet')
        # self.df_meta_song_lyricist = pd.read_parquet(f'data/meta_song_lyricist.parquet')
        # self.df_meta_song_producer = pd.read_parquet(f'data/meta_song_producer.parquet')
        # self.df_meta_song_titletext = pd.read_parquet(f'data/meta_song_titletext.parquet')

        # with open('data/json/training_source.json', 'r', encoding='utf-8') as file:
        #     self.training_source = json.load(file)
        
        with open('data/json/training_target.json', 'r', encoding='utf-8') as file:
            self.training_target = json.load(file)

        with open('data/json/cb.json', 'r', encoding='utf-8') as file:
            self.cb = json.load(file)

        with open('data/json/test_source.json', 'r', encoding='utf-8') as file:
            self.test_source = json.load(file)

        with open('data/json/test_target.json', 'r', encoding='utf-8') as file:
            self.test_target = json.load(file)

        with open('data/json/selected_songs.json', 'r', encoding='utf-8') as file:
            self.selected_songs = json.load(file)
        
        with open('data/json/similarity.json', 'r', encoding='utf-8') as file:
            self.similarity = json.load(file)

        # with open('data/json/analyzed_result.json', 'r', encoding='utf-8') as file:
        #     self.analyzed_result = json.load(file)
        
        # with open('data/json/analyzed_song_data.json', 'r', encoding='utf-8') as file:
        #     self.analyzed_song_data = json.load(file)

        # with open('data/json/song_id.json', 'r', encoding='utf-8') as file:
        #     self.song_id = json.load(file)

    def traverse_training_data(self):
        for key in self.training_source.keys():
            # self.analyzed_song_data[key] = dict()
            # self.check_multiply(key)

            print(key)

        self.write_result()

    def check_multiply(self,session_id):
        songs_list = self.training_source[session_id]
        target_list = self.training_target[session_id]
        cnt_songs = Counter(songs_list)
        cnt_target = Counter(target_list)

        self.analyzed_song_data[session_id]['num_songs'] = len(songs_list)

        # 判斷重複
        # 如果 session 中全部的歌都是同一首
        if cnt_songs.most_common(1)[0][1] == cnt_songs.total():
            self.session_all_same(session_id)

        # 如果 session 中有歌曲重複
        elif cnt_songs.most_common(1)[0][1] > 1 :
            self.session_song_repeat(session_id)

        # 如果 session 中都沒有歌曲重複
        else:
            self.analyzed_song_data[session_id]['is_allsame'] = False
            self.analyzed_song_data[session_id]['is_repeat'] = False
        

    """
    session 中全部的歌都是同一首
    input : (string) session_id
    """
    def session_all_same(self,session_id):
        songs_list = self.training_source[session_id]
        target_list = self.training_target[session_id]
        cnt_target = Counter(target_list)

        self.analyzed_song_data[session_id]['is_allsame'] = True
        self.analyzed_song_data[session_id]['is_repeat'] = True
        self.analyzed_song_data[session_id]['most_items'] = None
        self.analyzed_song_data[session_id]['song_in_target'] = cnt_target[songs_list[0]]


    """
    session 中有歌曲重複
    input : (string) session_id
    """
    def session_song_repeat(self,session_id):
        songs_list = self.training_source[session_id]
        target_list = self.training_target[session_id]
        cnt_songs = Counter(songs_list)
        cnt_target = Counter(target_list)

        self.analyzed_song_data[session_id]['is_allsame'] = False
        self.analyzed_song_data[session_id]['is_repeat'] = True

        # 排名前五首的狀況
        most_items_target = [0] * 5
        most_items = cnt_songs.most_common(5)
        for i in range(0,len(most_items)):
            if most_items[i][1]<=1:
                most_items[i] = None
                most_items_target[i] = None
            else:
                most_items_target[i] = cnt_target[most_items[i][0]]

        self.analyzed_song_data[session_id]['most_items'] = most_items
        self.analyzed_song_data[session_id]['most_items_target'] = most_items_target

        # 最尾端是否有重複
        self.analyzed_song_data[session_id]['repeat_at_end'] = False

        count = 0
        for i in range(len(songs_list)-1,-1,-1):
            if songs_list[i] == songs_list[-1]:
                count += 1
            else:
                break

        if count > 1:
            self.analyzed_song_data[session_id]['repeat_at_end'] = count
            self.analyzed_song_data[session_id]['repeat_at_end_target'] = cnt_target[songs_list[-1]]

    
    def count(self):
        num_is_allsame=0
        num_is_repeat=0
        num_is_allsame_and_song_in_target=0
        num_repeat_items = dict()
        num_repeat_at_end = dict()

        for key in self.analyzed_song_data.keys():
            print(key)
            if self.analyzed_song_data[key]['is_allsame']==True:
                num_is_allsame+=1
                if self.analyzed_song_data[key]['song_in_target'] > 0:
                    num_is_allsame_and_song_in_target+=1

            elif self.analyzed_song_data[key]['is_repeat']==True:
                num_is_repeat+=1
                tmp_list = self.analyzed_song_data[key]['most_items']
                tmp_list_target = self.analyzed_song_data[key]['most_items_target']
                for i in range(0,len(tmp_list)):
                    if tmp_list[i] == None: break
                    tmp = tmp_list[i][1]
                    if tmp not in num_repeat_items:
                        num_repeat_items[tmp]={'source':0,'target':0}
                    num_repeat_items[tmp]['source'] += 1
                    if tmp_list_target[i] > 0:
                      num_repeat_items[tmp]['target'] += 1

                if self.analyzed_song_data[key]['repeat_at_end']!=False:
                    tmp = self.analyzed_song_data[key]['repeat_at_end']
                    if tmp not in num_repeat_at_end:
                        num_repeat_at_end[tmp]={'source':0,'target':0}
                    num_repeat_at_end[tmp]['source']+=1
                    if self.analyzed_song_data[key]['repeat_at_end_target'] > 0:
                        num_repeat_at_end[tmp]['target']+=1

        self.analyzed_result['num_is_allsame'] = num_is_allsame
        self.analyzed_result['num_is_allsame_and_song_in_target'] = num_is_allsame_and_song_in_target
        self.analyzed_result['num_is_repeat'] = num_is_repeat
        self.analyzed_result['num_repeat_items'] = num_repeat_items
        self.analyzed_result['num_repeat_at_end'] = num_repeat_at_end
        self.write_result()

    
    def record_selected_song(self, list):
        for song in list:
            if song not in self.selected_songs.keys():
                self.selected_songs[song]=1

    def write_result(self):

        # with open('data/json/analyzed_result.json', 'w', encoding='utf-8') as file:
        #     json.dump(self.analyzed_result, file, indent=2)

        # with open('data/json/analyzed_song_data.json', 'w', encoding='utf-8') as file:
        #     json.dump(self.analyzed_song_data, file, indent=2)

        with open('data/json/selected_songs.json', 'w', encoding='utf-8') as file:
            json.dump(self.selected_songs, file, indent=2)

        with open('data/json/test_target.json', 'w', encoding='utf-8') as file:
            json.dump(self.test_target, file, indent=2)
        
        tmp = []
        for key, value in self.test_target.items():
            tmp.append({'session_id':key,'top1':value[0],'top2':value[1],'top3':value[2],'top4':value[3],'top5':value[4]})

        df = pd.json_normalize(tmp)
        df.to_csv('output_file1.csv', index=False, header=True, columns=['session_id','top1','top2','top3','top4','top5'])

    def reset_target_json(self):
        self.test_target = {}
        for key in self.test_source.keys():
            self.test_target[key]=[None]*5
        self.write_result()

    def training_data_to_json(self, df_source):
        training_data = {}

        for index, row in df_source.iterrows():
            session_id = row['session_id']
            song_id = row['song_id']
            if session_id not in training_data:
                training_data[session_id] = []  
            
            training_data[session_id].append(song_id)
            print(f"session id: {session_id}   song id: {song_id}")

        with open('data/test_source.json','w') as json_file:
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

    def song_id_to_json(self):
        df_dict = dict.fromkeys(self.df_meta_song['song_id'], 0)

        with open('data/json/song_id.json', 'w', encoding='utf-8') as file:
            json.dump(df_dict, file, indent=2)
