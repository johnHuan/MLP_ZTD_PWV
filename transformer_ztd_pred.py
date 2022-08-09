# -*- coding: utf-8 -*-
# @Time    : 2022/4/14 10:25
# @Author  : Zhang Huan
# @Email   : johnhuan@whu.edu.cn
# QQ       : 248404941
# @File    : transformer_ztd_train.py
import math
import os
import time

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

from sklearn.preprocessing import MinMaxScaler


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # .squeeze 对数据的维度进行压缩或者解压
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self.generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


#  if window in 100 and prediction step is 1
#  in       -> [0...99]
#  target   -> [1...100]
def create_in_out_sequences(input_data, in_window):
    in_out_seq = []
    L = len(input_data)
    for i in range(L - in_window):
        train_seq = input_data[i:i + in_window]
        train_label = input_data[i + output_window: i + output_window + in_window]
        in_out_seq.append((train_seq, train_label))
    # convert train data into a pytorch train tensor
    return torch.FloatTensor(in_out_seq)


def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))  # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return input, target


def train(train_data):
    model.train()  # turn on the train model \o/
    total_loss = 0.
    start_time = time.time()
    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.7)
        optimizer.step()
        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            current_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | {:5.2f} ms | loss {:5.5f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0], elapsed * 1000 / log_interval,
                current_loss, math.exp(current_loss)
            ))
            total_loss = 0
            start_time = time.time()


def plot_and_loss(eval_model, data_source, epoch, scaler, station):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1)
            output = eval_model(data)
            total_loss += criterion(output, target).item()
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)
    len(test_result)
    test_result = scaler.inverse_transform(test_result.reshape(-1, 1))
    truth = scaler.inverse_transform(truth.reshape(-1, 1))
    df = pd.DataFrame({'pred': np.array(test_result).reshape(-1), 'truth': np.array(truth).reshape(-1)})
    # df_prefix = pd.DataFrame(
    #     {'pred': np.array(test_result[:samples]).reshape(-1), 'truth': np.array(truth[:samples]).reshape(-1)})
    # df_sufix = pd.DataFrame(
    #     {'pred': np.array(test_result[samples:]).reshape(-1), 'truth': np.array(truth[samples:]).reshape(-1)})
    df.to_csv(target_dir + 'data/' + station[:-3] + 'data%d.csv' % epoch)
    # df_prefix.to_csv(target_dir + 'data/' + station[:-3] + 'data_prefix%d.csv' % epoch)
    # df_sufix.to_csv(target_dir + 'data/' + station[:-3] + 'data_sufix%d.csv' % epoch)
    plt.plot(test_result, color='red', label='pred')
    plt.plot(truth[:samples], color="blue", label='truth')
    plt.plot(test_result - truth, color="green", label='error')
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.legend(bbox_to_anchor=(.86, 1), loc=6, borderaxespad=0.06)
    plt.savefig(target_dir + 'fig/' + station[:-3] + 'transformer-epoch%d.png' % epoch)
    plt.close()
    return total_loss / i


def predict_future(eval_model, data_source, steps, scaler):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    data, target = get_batch(data_source, 0, 1)
    with torch.no_grad():
        for i in range(0, steps):
            output = eval_model(data[-input_window:])
            data = torch.cat((data, output[-1:]))
    data = data.cpu().view(-1)
    # I used this plot to visualize if the model pics up any long term structure within the data.
    data = scaler.inverse_transform(data.reshape(-1, 1))
    # TODO 这个data 和 output 才是 transformer模型出来的data
    print(data)
    print(output)
    # df = pd.DataFrame({'pred': np.array(test_result).reshape(-1), 'truth': np.array(truth[:samples]).reshape(-1)})
    df = pd.DataFrame({
        'output': np.array(output).reshape(-1),
        'data': np.array(data).reshape(-1)
    })
    df.to_csv('../transformer_model/ztd/data/data_%d.csv' % epoch)
    plt.plot(data, color="red", label='data')
    plt.plot(output, color="blue", label='output')
    # plt.plot(data, color="red", label='data')
    # plt.plot(data[:input_window], color="blue", label=data[:input_window])
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.legend(bbox_to_anchor=(.86, 1), loc=6, borderaxespad=0.06)
    plt.savefig('../transformer_model/ztd/fig/transformer-future%d.png' % steps)
    plt.close()


def evaluate(eval_model, data_source):
    eval_model.eval()  # turn on the evaluation model
    total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data)  # todo 耗时在这里
            total_loss += len(data[0]) * criterion(output, targets).cpu().item()
    return total_loss / len(data_source)


samples = 365 * 24 * 1
if __name__ == '__main__':

    torch.manual_seed(0)
    np.random.seed(0)
    # S is the source sequence length
    # T is the target sequence length
    # N is the batch size
    # E is the feature number

    # src =
    # tgt =
    # out =
    input_window = 100  # number of input steps
    output_window = 1  # number of prediction steps, in this model its fixed to one
    batch_size = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_dir = 'F:/ImplDoctor/marchineLearning/transformer/transformer_model/ztd/'
    base_dir = 'D:/lstm/gnss/lstm/period_model/used/'
    stations = os.listdir(base_dir + 'data/')
    stations.sort()
    stations.reverse()
    for station in stations:
        filepath = base_dir + 'data/' + station
        series = pd.read_csv(filepath, header=0, parse_dates=True, squeeze=True, usecols=['3'])
        scaler = MinMaxScaler(feature_range=(-1, 1))
        amplitude = scaler.fit_transform(series.to_numpy().reshape(-1, 1)).reshape(-1)
        cnt = amplitude.shape[0]
        cut = cnt - samples
        train_data = amplitude[cut:]
        test_data = amplitude[:cut]
        train_sequence = create_in_out_sequences(train_data, input_window)
        train_sequence = train_sequence[: -output_window]
        test_data = create_in_out_sequences(test_data, input_window)
        test_data = test_data[: -output_window]
        train_data = train_sequence.to(device)
        valid_data = test_data.to(device)
        model = TransAm().to(device)
        criterion = nn.MSELoss()
        lr = 0.005
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
        best_val_loss = float('inf')
        epochs = 54
        best_model = None
        epoch = epochs
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            train(train_data)
            # val_loss = evaluate(model, valid_data)
            # print('-' * 89)
            # print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'
            #       .format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss))
            #       )
            # print('-' * 89)
            scheduler.step()
        val_loss = plot_and_loss(model, valid_data, epochs, scaler, station)
        # 模型保存
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss
        }, target_dir + 'model/' + station[:-3] + '_model_%d.pkl' % epochs)

        # 模型加载
        """
        model = TheModelClass(*args, **kwargs)
        optimizer = TheOptimizerClass(*args, **kwargs)

        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        model.eval()
        # - or -
        model.train()
        """
        # predict_future(model, valid_data, epochs, scaler)
