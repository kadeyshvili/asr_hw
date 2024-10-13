
import torch
import torch.nn as nn
import torch.nn.functional as F


class RnnReluBlock(nn.Module):
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,
    }

    def __init__(
            self,
            input_size: int,
            hidden_size: int = 512,
            rnn_type: str = 'gru',
            bidirectional: bool = True,
            dropout: float = 0.1,
    ):
        super(RnnReluBlock, self).__init__()
        self.hidden_size = hidden_size
        self.batch_norm = nn.BatchNorm1d(input_size)
        rnn_cell = self.supported_rnns[rnn_type]
        self.rnn = rnn_cell(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, inputs, input_lengths):
        inputs = inputs.permute(0, 2, 1) #(batch_size, hidden_size, seq_len)
        inputs = F.relu(self.batch_norm(inputs)) 
        inputs = inputs.permute(0, 2, 1) #(batch_size,seq_len,  hidden_size )

        outputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths.cpu(), enforce_sorted=False, batch_first=True)
        outputs, hidden_states = self.rnn(outputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        if self.rnn.bidirectional:
            outputs = outputs.view(outputs.size(0), outputs.size(1), 2, -1).sum(2).view(outputs.size(0), outputs.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum [2, 382, 2048]-> [2, 382, 1024]
        return outputs

class DeepSpeech2(nn.Module):

    def __init__(
        self,
        n_feats: int,
        n_tokens: int, 
        hidden_size: int = 512,
        num_rnn_layers: int = 5,
        dropout: float = 0.3,
        bidirectional: bool = True,
        rnn_type: str = 'gru',
    ):
        super(DeepSpeech2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (41, 11), stride=(2, 2), padding=(20, 5), bias=False),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, (21, 11), stride=(2, 1), padding=(10, 5), bias=False),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 96, (21, 11), stride=(2, 1), padding=(10, 5), bias=False),
        )

        input_size = n_feats
        input_size = (input_size + 20 * 2 - 41) // 2 + 1
        input_size = (input_size + 10 * 2 - 21) // 2 + 1
        input_size = (input_size + 10 * 2 - 21) // 2 + 1
        input_size *= 96


        self.rnn = nn.ModuleList()
        for i in range(num_rnn_layers):
            self.rnn.append(
                RnnReluBlock(
                    input_size=input_size if i == 0 else hidden_size,
                    hidden_size=hidden_size,
                    rnn_type=rnn_type,
                    bidirectional=bidirectional,
                    dropout=dropout,
                )
            )
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        spectrogram = spectrogram.unsqueeze(1)

        x = self.conv(spectrogram)

        batch, seq_len, num_ch, hidden_dim = x.shape

        seq_lens = spectrogram_length 
        for module in self.conv:
            if isinstance(module, nn.Conv2d):
                seq_lens = seq_lens + (2 * module.padding[1]) - module.dilation[1] * (module.kernel_size[1] - 1) - 1
                seq_lens = seq_lens.float() / float(module.stride[1])
                seq_lens = seq_lens.int() + 1


        x = x.view(batch, seq_len * num_ch, hidden_dim).transpose(1, 2)
        for layer in self.rnn:
            x = layer(x, seq_lens)

        x = self.batch_norm(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        log_probs = F.log_softmax(self.fc(x), dim=-1) 

        return {"log_probs": log_probs, "log_probs_length": seq_lens}
