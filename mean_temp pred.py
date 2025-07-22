import S_LSTM_train
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

df_t = pd.read_csv('/Users/liangke/Desktop/Kaggle_Datasets/daily-climate-time-series-data/DailyDelhiClimateTest.csv')

model_final = S_LSTM_train.LSTM(6, 128, 2, 1)
model_final.load_state_dict(torch.load('Climate_LSTM.pth'))
model_final.eval()

#预处理
df_t['Z_temp'] = (df_t['meantemp'] - S_LSTM_train.t_mean) / S_LSTM_train.t_std
df_t['Z_pressure'] = (df_t['meanpressure'] - S_LSTM_train.p_mean) / S_LSTM_train.p_std
df_t['Z_hum'] = (df_t['humidity'] - S_LSTM_train.h_mean) / S_LSTM_train.h_std
df_t['Z_wspeed'] = (df_t['wind_speed'] - S_LSTM_train.w_mean) / S_LSTM_train.w_std

cs_mon = df_t['date'].apply(S_LSTM_train.get_date)
df_t['mon_sin'] = np.sin(np.pi * 2 * cs_mon / 12)
df_t['mon_cos'] = np.cos(np.pi * 2 * cs_mon / 12)

X_tst = torch.tensor(df_t[S_LSTM_train.input_character].values, dtype = torch.float32)
y_tst = torch.tensor(df_t['Z_temp'].values, dtype = torch.float32)

X_tst_seq, y_tst_seq = S_LSTM_train.creat_sequences(X_tst, y_tst, S_LSTM_train.seq_len, len(df_t))

tst_dataset = TensorDataset(X_tst_seq, y_tst_seq)

tst_dataloader = DataLoader(dataset = tst_dataset, batch_size = S_LSTM_train.batch,
                            shuffle = False)


y_pred = []
with torch.no_grad():
    for X_batch, _ in tst_dataloader:
        output_final = model_final(X_batch)
        y_pred.append(output_final.numpy())

y_pred = np.concatenate(y_pred).flatten()
y_pred = y_pred * S_LSTM_train.t_std + S_LSTM_train.t_mean

plt.plot(df_t['date'].iloc[7:], df_t['meantemp'].iloc[7:], color = 'skyblue', label = 'true temp')
plt.plot(df_t['date'].iloc[7:], y_pred, color = 'green', label = 'pred temp')
plt.xticks(range(0, 114, 15), rotation = 45)
plt.legend()
plt.show()