import numpy as np
import pandas as pd

from utils.data_tidy import UserData, SongData
from utils.model import NCF, SongLensTrainDataset

from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn

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

    song_id = song.df_meta_song
    song_id = song_id[['song_id']]
    all_song_id = song_id['song_id'].unique()

    num_user = len(dataset['session_id'].unique())
    num_song = len(all_song_id)

    print(num_user, num_song)

    ############################################################################

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    model = NCF(num_user, num_song).to(device)

    data = DataLoader(
        SongLensTrainDataset(dataset, all_song_id), batch_size=1024, num_workers=4
    )

    num_epochs = 200

    NCF_opt = torch.optim.Adam(model.parameters())

    for epoch in range(num_epochs):

        print(f"epoch: {epoch}")

        for batch_idx, batch in enumerate(data):

            user_input, item_input, labels = batch
            user_input, item_input, labels = (
                user_input.to(device),
                item_input.to(device),
                labels.to(device),
            )

            predicted_labels = model(user_input, item_input)

            loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())

            loss.backward()
            NCF_opt.step()

            if batch_idx % len(data) == 500:

                print(f"batch: {batch_idx}")

    checkpoint = {"state_dict": model.state_dict(), "optimizer": NCF_opt.state_dict()}

    torch.save(model, "./NCF_trained.pth")
