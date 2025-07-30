import torch.nn as nn
import torch 

# define the RNN 
class RNN(nn.Module):
  
  def __init__(self , vocab_size , embedding_dim , hidden_dim , output_dim , pad_index):
      super().__init__()

      self.embedding = nn.Embedding(vocab_size , embedding_dim , padding_idx=pad_index)
      self.dropout1 = nn.Dropout(0.4)
      self.rnn = nn.LSTM(embedding_dim , hidden_dim , batch_first=True , bidirectional=True)
      self.dropout2 = nn.Dropout(0.4)

      # The hidden dimension of the linear layer should be 2 * hidden_dim because of the bidirectional LSTM
      self.fc = nn.Linear(hidden_dim * 2 , output_dim)

  def forward(self , x):
    embedded = self.embedding(x)
    embedded = self.dropout1(embedded) # Apply dropout after embedding

    _ , (hidden , _) = self.rnn(embedded)
    
    # hidden has shape (2, batch_size, hidden_dim) for bidirectional LSTM
    # Concatenate the hidden states from the forward and backward directions
    hidden = torch.cat((hidden[0,:,:], hidden[1,:,:]), dim=1)
    hidden = self.dropout2(hidden) # Apply dropout before the linear layer
    return self.fc(hidden)
  