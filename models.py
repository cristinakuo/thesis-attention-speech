import torch.nn as nn

class ASR(nn.Module):

    def __init__(self, input_size, output_size):
        super(ASR, self).__init__()

        # Encoder
        hidden_size = 100 # Random
        num_layers = 1
        self.encoder = Encoder(input_size, hidden_size, num_layers, output_size)

    def forward(self, input_x):
        return self.encoder(input_x)

class Encoder(nn.Module):
    ''' Encoder (a.k.a. Listener in LAS)
        Encodes acoustic feature to latent representation, see config file for more details.'''

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Encoder, self).__init__()

        # Construct model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # Recurrent encoder # Maybe just 1 layer for NOW, we can use a simple BLSTM and then pass the layers
        self.blstm_layer = nn.LSTM(self.input_size, self.hidden_size, self.num_layers,
                                    batch_first=True, bidirectional=True, bias=True)

    def forward(self, input_x):
        # TODO: manage batches

        return self.blstm_layer(input_x)