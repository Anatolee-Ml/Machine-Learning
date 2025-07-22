import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#import matplotlib.pyplot as plt

df = pd.read_csv("/Users/liangke/Desktop/Kaggle_Datasets/daily-climate-time-series-data/DailyDelhiClimateTrain.csv")
#1462

batch = 32
seq_len = 7
input_character = ['Z_temp', 'Z_pressure', 'Z_wspeed', 'Z_hum', 'mon_sin', 'mon_cos']

#类、函数
def get_date(date):
    year, month, day = date.split('-')
    return int(month)

def creat_sequences(data, target, seq_len, len):
    X_seq = []
    y_seq = []
    for i in range(len - seq_len):
        X_seq.append(data[i : i + seq_len])
        y_seq.append(target[i + seq_len])
    return torch.stack(X_seq), torch.stack(y_seq)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=False
        )

        self.linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x



#可视化
'''
plt.figure(figsize = (12, 8))

plt.subplot(2, 1, 1)
plt.plot(df['date'], df['meantemp'], color = 'skyblue')
plt.xticks(range(0, 1463, 365))

plt.subplot(2, 1, 2)
plt.plot(df['date'], df['humidity'], color = 'r')
plt.xticks(range(0, 1463, 365))

plt.show()
'''

#Z分数
#==================================================
p_mean = df['meanpressure'].mean()
p_std = df['meanpressure'].std()
#print(p_mean, p_std, sep = "    ")
df['Z_pressure'] = (df['meanpressure'] - p_mean) / p_std
df.loc[df['Z_pressure'] > 0.3, 'Z_pressure'] = 0.3
df.loc[df['Z_pressure'] < -0.3, 'Z_pressure'] = -0.3
'''
print(df['Z_pressure'].describe())
plt.boxplot(df['Z_pressure'])
plt.show()
'''

w_mean = df['wind_speed'].mean()
w_std = df['wind_speed'].std()
#print(w_mean, w_std, sep = "    ")
df['Z_wspeed'] = (df['wind_speed'] - w_mean) / w_std
df.loc[df['Z_wspeed'] > 3, 'Z_wspeed'] = 3
df.loc[df['Z_wspeed'] < -3, 'Z_wspeed'] = -3
'''
print(df['Z_wspeed'].describe())
plt.boxplot(df['Z_wspeed'])
plt.show()
'''

t_mean = df['meantemp'].mean()
t_std = df['meantemp'].std()
h_mean = df['humidity'].mean()
h_std = df['humidity'].std()

df['Z_temp'] = (df['meantemp'] - t_mean) / t_std
df['Z_hum'] = (df['humidity'] - h_mean) / h_std
#==================================================

#月份编码（sin、cos）
new_month = df['date'].apply(get_date)
df['mon_sin'] = np.sin(2 * np.pi * new_month / 12)
df['mon_cos'] = np.cos(2 * np.pi * new_month / 12)

print(df.columns)


#训练+保存
if __name__  == '__main__':

    # 制作数据ji

    X = torch.tensor(df[input_character].values, dtype=torch.float32)
    y = torch.tensor(df['Z_temp'].values, dtype=torch.float32).reshape(-1, 1)

    X, y = creat_sequences(X, y, seq_len, len(df))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=False)

    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)

    train_dataloader = DataLoader(dataset=train_data, batch_size=batch,
                                  shuffle=True)
    val_dataloader = DataLoader(dataset=val_data, batch_size=batch,
                                shuffle=False)


    MODEL = LSTM(6, 128, 2, 1)

    criterion = nn.MSELoss()
    opt = optim.Adam(MODEL.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5)

    min_loss = np.inf
    patient = 20
    p_count = 0
    epoch = 100
    for i in range(epoch):
        MODEL.train()

        for X_batch, y_batch in train_dataloader:
            output = MODEL(X_batch)
            loss = criterion(output, y_batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

        MODEL.eval()

        with torch.no_grad():
            total_loss = 0
            pred = []
            label = []
            for X_batch, y_batch in val_dataloader:
                output = MODEL(X_batch)
                val_loss = criterion(output, y_batch)
                pred.append(output.numpy())
                label.append(y_batch.numpy())

                total_loss += val_loss.item()

        pred = np.concatenate(pred).flatten()
        label = np.concatenate(label).flatten() #展平为一维数组（向量）

        r2 = r2_score(label, pred)

        scheduler.step(total_loss)
        if not (i + 1) % 3:
            print(f'[{i + 1} / {epoch}] Loss = {total_loss:.4f} ; r2 = {r2:.4f}')

        if total_loss < min_loss:
            min_loss = total_loss
            p_count = 0
            torch.save(MODEL.state_dict(), 'Climate_LSTM.pth')
        else:
            p_count += 1

        if p_count > patient:
            print('early stopped')
            break

'''
[3 / 100] Loss = 1.0135 ; r2 = 0.9254
[6 / 100] Loss = 1.1567 ; r2 = 0.9148
[9 / 100] Loss = 0.9171 ; r2 = 0.9326
[12 / 100] Loss = 1.0113 ; r2 = 0.9256
[15 / 100] Loss = 0.6816 ; r2 = 0.9502
[18 / 100] Loss = 0.7815 ; r2 = 0.9432
[21 / 100] Loss = 0.7041 ; r2 = 0.9485
[24 / 100] Loss = 0.7170 ; r2 = 0.9475
[27 / 100] Loss = 0.7064 ; r2 = 0.9483
[30 / 100] Loss = 0.7045 ; r2 = 0.9484
[33 / 100] Loss = 0.6978 ; r2 = 0.9489
[36 / 100] Loss = 0.6979 ; r2 = 0.9489
early stopped
'''