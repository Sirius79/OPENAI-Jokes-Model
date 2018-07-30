import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class RNN(nn.Module):
  '''
  Uses 3 layered bidirectional GRU
  '''
  def __init__(self, input_size, hidden_size, output_size):
    super(RNN, self).__init__()
    self.hidden_size = hidden_size
    
    self.embedding = nn.Embedding.from_pretrained(weights)
    self.gru = nn.GRU(hidden_size, hidden_size, num_layers=3, bidirectional=True)
    self.out = nn.Linear(hidden_size*2, output_size)
    self.softmax = nn.LogSoftmax(dim=2)
  
  def forward(self, input, hidden):
    embedded = self.embedding(input).view(1,1,-1)
    output = embedded
    output, hidden = self.gru(output, hidden)
    output = self.softmax(self.out(output))
    return output, hidden
  
  def initHidden(self):
    return torch.zeros(6, 1, self.hidden_size, device=device)
