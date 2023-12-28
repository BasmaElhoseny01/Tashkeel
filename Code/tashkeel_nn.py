from utils import *
from Globals import vocab_size,n_classes

class Tashkeel(nn.Module):
  def __init__(self, vocab_size=vocab_size, embedding_dim=50, hidden_size=50, n_classes=n_classes):
    """
    The constructor of our Tashkeel model
    Inputs:
    - vacab_size: the number of unique words
    - embedding_dim: the embedding dimension
    - n_classes: the number of final classes (tags)
    """
    super(Tashkeel, self).__init__()
    # (1) Create the embedding layer
    self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)

    # (2) Create an LSTM layer with hidden size = hidden_size and batch_first = True
    self.lstm =  nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size,batch_first=True)

    # (3) Create a linear layer with number of neorons = n_classes
    self.linear =  nn.Linear(hidden_size,n_classes)
    #####################################################################################################
  
  def forward(self, sentences):
      """
      This function does the forward pass of our model
      Inputs:
      - sentences: tensor of shape (batch_size, max_length)

      Returns:
      - final_output: tensor of shape (batch_size, max_length, n_classes)
      """
      final_output = None
      embedding=self.embedding(sentences)

      # output, (hidden_state, cell_state)
      lstm_out,_=self.lstm(embedding)

      linear_out=self.linear(lstm_out)

      final_output=linear_out
      ###############################################################################################
      return final_output