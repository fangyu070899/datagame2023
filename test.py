import numpy as np
import pandas as pd
import csv

from utils.data_tidy import UserData, SongData
from utils.model import NCF, SongLensTrainDataset

from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn

import scipy.sparse as sp


def get_recommendations(u, all_songIds):

    interacted_items = user_interacted_items[u]
    not_interacted_items = list(set(all_songIds) - set(interacted_items))

    predicted_labels = np.squeeze(
        model(
            torch.tensor([u] * len(not_interacted_items)),
            torch.tensor(not_interacted_items),
        )
        .detach()
        .numpy()
    )

    top5_items = [
        not_interacted_items[i]
        for i in np.argsort(predicted_labels)[::-1][0:5].tolist()
    ]

    # print(top5_items)

    # ncf_recs = dataset[dataset["song_index"].isin(top10_items)]

    # return ncf_recs["song_id"].head(10).tolist()
    return top5_items


def get_predict(top_items, s_dict):

    d = []
    for rec in top_items:
        d.append(s_dict[rec][0])

    return d


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if __name__ == '__main__':

    data = UserData()
    song = SongData()


    # 整理data
    train_source = data.df_label_train_source
    train_source = train_source[['session_id', 'song_id', 'listening_order']]
    # print(train_source.head(40))

    train_target = data.df_label_train_target
    train_target = train_target[['session_id', 'song_id', 'listening_order']]
    # print(train_target.head(40))

    test_source = data.df_label_test_source
    test_source = test_source[['session_id', 'song_id', 'listening_order']]
    # print(test_source.head(40))

    dataset = pd.concat([train_source, train_target, test_source], ignore_index=True)
    dataset = dataset.sort_values(by=['session_id', 'listening_order'], ascending=[True, True])
    dataset['song_index'] = dataset['song_id'].astype('category').cat.codes

    all_song_id = dataset['song_index'].unique()

    num_user = len(dataset['session_id'].unique())
    num_song = len(all_song_id)

    all_song_id = list(all_song_id)
   
    print(num_user, num_song)
    
    model = NCF(num_user, num_song).to(device)

    if device == torch.device("cuda"):
        checkpoint = torch.load("model_checkpoint.pth", map_location=device)
    else:
        checkpoint = torch.load("model_checkpoint.pth", map_location=torch.device("cpu"))

    print("done")

    model.load_state_dict(checkpoint["state_dict"])

    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()

    user_interacted_items = dataset.groupby("session_id")["song_index"].apply(list).to_dict()
    song_dict = dataset.groupby("song_index")["song_id"].unique().apply(list).to_dict()
    test = test_source['session_id'].unique()

##############################################################################
    
    # 寫入 CSV 文件
    csv_file_path = "song_dict.csv"
    with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # 標題行
        csv_writer.writerow(["session_id", "top1", "top2", "top3", "top4", "top5"])
        c = 0

        for id in test:
            c += 1
            if c % 10000 == 0 : print(c)

            resc = get_recommendations(id, all_song_id)
            sol = get_predict(resc, song_dict)

            csv_writer.writerow([id] + sol)

    print(f"CSV 文件已保存到: {csv_file_path}")
    
    # for key, value in song_dict.items():
    #     print(key, ":", value)
    #     break

    # print(test_source.loc[test_source.index[0], 'session_id'])
    # recs = get_recommendations(test_source.loc[test_source.index[0], 'session_id'], all_song_id)
   

    # print("Based on your previous watches, we recommend:")
    # print()
    # for rec in recs:
    #     print(rec, ":", song_dict[rec])
