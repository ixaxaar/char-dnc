# https://github.com/spro/char-rnn.pytorch

import torch as T
import torch.nn as nn

from dnc import *
from util import *


class CharLM(nn.Module):

  def __init__(
      self,
      input_size,
      hidden_size,
      output_size,
      rnn_type="gru",
      n_layers=1,
      nr_cells=10,
      read_heads=4,
      cell_size=32,
      gpu_id=-1
  ):
    super(CharLM, self).__init__()
    self.kind = rnn_type.lower()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_layers = n_layers
    self.nr_cells = nr_cells
    self.read_heads = read_heads
    self.cell_size = cell_size
    self.gpu_id = gpu_id

    self.encoder = nn.Embedding(input_size, hidden_size)
    if self.kind == "gru":
      self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
    elif self.kind == "lstm":
      self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
    elif self.kind == 'dnc':
      self.rnn = DNC(
          mode='lstm',
          hidden_size=hidden_size,
          num_layers=n_layers,
          nr_cells=nr_cells,
          read_heads=read_heads,
          cell_size=cell_size,
          batch_first=True,
          gpu_id=self.gpu_id,
          independent_linears=False
      )
      # register_nan_checks(self.rnn)
    self.decoder = nn.Linear(hidden_size, output_size)
    # register_nan_checks(self.decoder)

  def forward(self, input, hidden=None):
    batch_size = input.size(0)
    encoded = self.encoder(input)
    if self.kind == 'dnc':
      output, hidden = self.rnn(
          encoded.view(batch_size, 1, -1),
          hidden,
          reset_experience=self.reset_experience
      )
      output = output.transpose(0, 1)
    else:
      output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
    output = self.decoder(output.view(batch_size, -1))

    return output, hidden

  def forward2(self, input, hidden=None):
    encoded = self.encoder(input.view(1, -1))
    output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
    output = self.decoder(output.view(1, -1))
    return output, hidden
