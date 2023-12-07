## 只是把song的data整理成dict

import pandas as pd

class Data:
  def __init__(self) :

    self.df_label_train_source = pd.read_parquet(f'label_train_source.parquet')
    self.df_label_train_target = pd.read_parquet(f'label_train_target.parquet')

    self.df_meta_song = pd.read_parquet(f'meta_song.parquet')
    self.df_meta_song_composer = pd.read_parquet(f'meta_song_composer.parquet')
    self.df_meta_song_genre = pd.read_parquet(f'meta_song_genre.parquet')
    self.df_meta_song_lyricist = pd.read_parquet(f'meta_song_lyricist.parquet')
    self.df_meta_song_producer = pd.read_parquet(f'meta_song_producer.parquet')
    self.df_meta_song_titletext = pd.read_parquet(f'meta_song_titletext.parquet')


data = Data()

print(data.df_label_train_source.keys())

# 甚麼東西不想要自己選
keys_to_drop = ['unix_played_at', 'play_status', 'login_type']
data.df_label_train_source = data.df_label_train_source.drop(keys_to_drop, axis=1)
# print(data.df_label_train_source.head(40))


# 知道每個檔案給什麼keys
print("song")
print(data.df_meta_song.keys())
print("song composer")
print(data.df_meta_song_composer.keys())
print("song genre")
print(data.df_meta_song_genre.keys())
print("song lyricist")
print(data.df_meta_song_lyricist.keys())
print("song producer")
print(data.df_meta_song_producer.keys())
print("song titletext")
print(data.df_meta_song_titletext.keys())


# 建立song的dict() & 紀錄song id
song_list = dict()
song_id_list = []

for index, row in data.df_meta_song.iterrows():
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
for index, row in data.df_meta_song_composer.iterrows():
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
for index, row in data.df_meta_song_genre.iterrows():
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
for index, row in data.df_meta_song_lyricist.iterrows():
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
for index, row in data.df_meta_song_producer.iterrows():
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
for index, row in data.df_meta_song_titletext.iterrows():
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

# test
# for i in range(0,30):
#     print(song_list[song_id_list[i]])