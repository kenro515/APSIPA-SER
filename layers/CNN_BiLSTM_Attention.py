import math
import torch
import torch.nn as nn

from .MultiheadAttention import MultiheadAttention

torch.manual_seed(1234)

class CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, d_model=None):
        super(CNN_BiLSTM_Attention, self).__init__()

        self.conv_layer1 = nn.Conv2d(
            1,
            64,
            kernel_size=5,
            stride=1
        )
        self.maxpool_layer1 = nn.MaxPool2d(
            kernel_size=4,
            stride=4
        )
        self.conv_layer2 = nn.Conv2d(
            64,
            128,
            kernel_size=5,
            stride=1
        )

        self.maxpool_layer2 = nn.MaxPool2d(
            kernel_size=4,
            stride=4
        )

        self.linear_layer1 = nn.Linear(
            5, 
            1
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=(d_model // 2),
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )

        self.mh_attention = MultiheadAttention(
            d_model=d_model
        )

        self.linear_out = nn.Linear(
            d_model,
            4
        )

        self.activation = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.5)

        self.feature_ = None

    def forward(self, x_ins):
        output_list = []
        for x_in in x_ins:
            if torch.nonzero(x_in, as_tuple=True)[0].shape[0] == 0:
                continue
            else:
                x_cnn1 = self.conv_layer1(x_in)
                x_cnn1_act = self.activation(x_cnn1)
                x_pool1 = self.maxpool_layer1(x_cnn1_act)
                
                x_cnn2 = self.conv_layer2(x_pool1)
                x_cnn2_act = self.activation(x_cnn2)
                x_pool2 = self.maxpool_layer2(x_cnn2_act)
                
                x_linear = self.linear_layer1(x_pool2)
                x_drop = self.dropout(x_linear)
                x_drop = x_drop.view(x_drop.shape[0], 1, x_drop.shape[1])
                output_list.append(x_drop)

        cnn2rnn = output_list[0]
        for cat_tenor in output_list[1:]:
            cnn2rnn = torch.cat([cnn2rnn, cat_tenor], dim=1)

        x_lstm, _ = self.lstm(cnn2rnn)
        self.feature_ = x_lstm

        input_mask = (x_lstm[:, :, 0] != 0)
        input_mask = input_mask.int()
        x_attn, _ = self.mh_attention(x_lstm, x_lstm, x_lstm, input_mask)
        x_attn_sum = x_attn.sum(dim=1)
        x_attn_drop = self.dropout(x_attn_sum)
        x_out = self.linear_out(x_attn_drop)

        return x_out, self.feature_

