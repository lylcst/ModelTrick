from transformers import ErnieModel, BertTokenizer
import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np

# LSTM Pooling
class LSTMPooling(nn.Module):
    def __init__(self, input_size=768, hidden_size=768):
        super(LSTMPooling, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)

    def forward(self, all_hidden_states):
        num_layers = len(all_hidden_states)
        hidden_states = torch.stack(
            [all_hidden_states[layer][:,0] for layer in range(1, num_layers)], dim=-1
        ).view(-1, num_layers-1, self.input_size)

        out, _ = self.lstm(hidden_states, None)
        return out[:, -1]


# Mean Pooling
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

# Max pooling
class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()
    
    def forward(self, last_hidden_state):
        bz, seq_len, hidden_dim = last_hidden_state.shape
        last_hidden_state = last_hidden_state.permute(0, 2, 1)
        out = torch.max_pool1d(last_hidden_state, seq_len, stride=seq_len)
        return out.permute(0, 2, 1).view(bz, hidden_dim)


class LastFourLayerConcatPooling(nn.Module):
    def __init__(self):
        super(LastFourLayerConcatPooling, self).__init__()
        self.fc = nn.Linear(4*768, 768)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, all_hidden_states):
        last_four_cls = torch.cat([layer[:, 0] for layer in all_hidden_states[-4:]], dim=-1)
        out = self.dropout(self.relu(self.fc(last_four_cls)))
        return out


class LastFourLayerMeanPooling(nn.Module):
    def __init__(self):
        super(LastFourLayerMeanPooling, self).__init__()
    
    def forward(self, all_hidden_states):
        out = torch.cat([layer[:, 0] for layer in all_hidden_states[-4:]], dim=-1).view(-1, 4, 768)
        out = torch.mean(out, dim=1)
        return out

class LastFourLayerMaxPooling(nn.Module):
    def __init__(self):
        super(LastFourLayerMaxPooling, self).__init__()
    
    def forward(self, all_hidden_states):
        out = torch.cat([layer[:, 0] for layer in all_hidden_states[-4:]], dim=-1).view(-1, 4, 768)
        out, _ = torch.max(out, dim=1)
        return out



# AttentionPooling
# class AttentionPooling(nn.Module):
#     def __init__(self):
#         super(AttentionPooling, self).__init__()
#         self.conv1 = nn.Conv1d(768, 256, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv1d(768, 256, kernel_size=5, stride=1, padding=2)
#         self.conv3 = nn.Conv1d(768, 256, kernel_size=7, stride=1, padding=3)
#         self.lstm = nn.LSTM(768, 768, num_layers=2, batch_first=True, bidirectional=True)
#         self.fc1 = nn.Linear(768*2, 768)
#         self.fc2 = nn.Linear(768*2, 768)
#         self.dropout = nn.Dropout(0.3)
    
#     def forward(self, last_hidden_state):
#         x = last_hidden_state.permute(0, 2, 1)
#         conv1_out = self.conv1(x) # [bz, 256, seq_len]
#         conv2_out = self.conv2(x) # [bz, 256, seq_len]
#         conv3_out = self.conv3(x) # [bz, 256, seq_len]
#         conv_out = torch.cat([conv1_out, conv2_out, conv3_out], dim=1) # [bz, 768, seq_len]
#         lstm_out, _ = self.lstm(last_hidden_state, None) # [bz, seq_len, hidden_size*2]
#         lstm_h = self.dropout(self.fc1(lstm_out[:, -1])) #[bz, hidden_size]
#         lstm_h = lstm_h.unsqueeze(1) #[bz, 1, hidden_size]
#         attn_weight = F.softmax(torch.matmul(lstm_h, conv_out), dim=-1) # [bz, 1, seq_len]
#         out = torch.matmul(attn_weight, conv_out.permute(0, 2, 1)).squeeze(1) # [bz, hidden_size]
#         # 拼接lstm_h
#         out = torch.cat([out, lstm_h.squeeze(1)], dim=1) # [bz, hidden_size*2]
#         out = self.dropout(self.fc2(out))
#         return out

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size=768, hidden_dim_fc=768):
        super(AttentionPooling, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_dim_fc = hidden_dim_fc
        self.dropout = nn.Dropout(0.1)

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        self.q = nn.Parameter(torch.from_numpy(q_t).float())
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.hidden_dim_fc))
        self.w_h = nn.Parameter(torch.from_numpy(w_ht).float())

    def forward(self, last_hidden_state):
        # [bz, seq_len, hidden_size]
        v = torch.matmul(self.q, last_hidden_state.transpose(-2, -1)).squeeze(1) # [bz, seq_len]
        v = F.softmax(v, -1) # [bz, seq_len]
        v_temp = torch.matmul(v.unsqueeze(1), last_hidden_state).transpose(-2, -1) #[bz, 1, seq_len]  [bz, seq_len, hidden_size] -> [bz, 1, hidden_size] -> [bz, hidden_size, 1]
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2) #[hidden_dim_fc, hidden_size] [bz, hidden_size, 1] -> [bz, hidden_dim_fc, 1] -> [bz, hiddden_fim_fc]
        return v



class LstmLayer(nn.Module):
    def __init__(self, input_size=768, hidden_size=768, output_dim=768, layer_num=1, bidirectional=True):
        super(LstmLayer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layer_num, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(hidden_size*2, output_dim)
        else:
            self.fc = nn.Linear(hidden_size, output_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        out, _ = self.lstm(x, None)
        out = self.dropout(self.fc(out[:, -1]))
        return out

class TextCNN(nn.Module):
    def __init__(self, in_channels=768, out_channels=256, hidden_size=768):
        super(TextCNN, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=1)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=1)
        self.fc = nn.Linear(3 * self.out_channels, hidden_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: [bz, seq_len, hidden_size] -> [bz, hidden_size, seq_len]
        bz, seq_len, hidden_dim = x.shape
        x = x.permute(0, 2, 1)
        cnn1 = self.conv1(x)
        cnn1 =  nn.MaxPool1d(kernel_size=cnn1.size(2), stride=cnn1.size(2))(cnn1).permute(0, 2, 1).view(bz, self.out_channels)
        cnn2 = self.conv2(x)
        cnn2 =  nn.MaxPool1d(kernel_size=cnn2.size(2), stride=cnn2.size(2))(cnn2).permute(0, 2, 1).view(bz, self.out_channels)
        cnn3 = self.conv3(x)
        cnn3 =  nn.MaxPool1d(kernel_size=cnn3.size(2), stride=cnn3.size(2))(cnn3).permute(0, 2, 1).view(bz, self.out_channels)
        out = torch.cat([cnn1, cnn2, cnn3], dim=-1)
        out = self.dropout(self.fc(out))
        return out

class RCNNLayer(nn.Module):
    def __init__(self, input_size=768, hidden_size=768, output_size=768, layer_num=1):
        super(RCNNLayer, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=layer_num, batch_first=True, bidirectional=True)

    def forward(self, x):
        out, _ = self.lstm(x, None)
        return out.chunk(2, dim=-1)


class DPCNNLayer(nn.Module):
    def __init__(self, num_filters=250, hidden_size=768):
        super(DPCNNLayer, self).__init__()
        self.conv_region = nn.Conv2d(1, num_filters, (3, hidden_size), stride=1)
        self.conv = nn.Conv2d(num_filters, num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(num_filters, hidden_size)

    def forward(self, x):
        # TODO DPCNN
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, embed]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] >= 2:
            x = self._block(x)
        x = x.squeeze(2).squeeze(2) # [batch_size, num_filters(250)]
        x = self.fc(x)
        return x

    def _block(self, x):
        x = self.padding2(x) # [batch_size, 250, seq_len-1, 1]
        px = self.max_pool(x) # [batch_size, 250, seq_len/2-2, 1]
        x = self.padding1(px) # [batch_size, 250, seq_len/2, 1]
        x = F.relu(x)
        x = self.conv(x) # [batch_size, 250, seq_len/2-3+1, 1]
        x = self.padding1(x) # [batch_size, 250, seq_len/2, 1]
        x = F.relu(x)
        x = self.conv(x)# [batch_size, 250, seq_len/2-2, 1]
        x = x + px  # short cut
        return x


class GenerateModel(torch.nn.Module):
    def __init__(self, model_path, istrain=True):
        super().__init__()
        self.istrain = istrain
        self.model =  ErnieModel.from_pretrained(model_path)
        self.pooler = LastFourLayerMaxPooling()
        self.loss_fct = torch.nn.BCELoss()
        self.predictor= torch.nn.Linear(768, 1)

    def forward(self,input_ids,token_type_ids,attention_mask,  cls_labels=None):

        encode_result = self.model(input_ids=input_ids, 
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            return_dict=True
                        )
        
        hidden_states = self.pooler(encode_result.hidden_states)
        predict_logits = torch.squeeze(self.predictor(hidden_states), dim=-1)

        if self.istrain:
            cls_loss = self.loss_fct(torch.sigmoid(predict_logits), cls_labels)
            return torch.mean(cls_loss)
        else:
            return torch.sigmoid(predict_logits)
