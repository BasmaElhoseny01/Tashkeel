from utils import *
from Globals import vocab_size,n_classes

class Tashkeel(nn.Module):
  def __init__(self, vocab_size=vocab_size, embedding_dim=512, hidden_size=256, n_classes=n_classes):
    """
    The constructor of our Tashkeel model
    Inputs:
    - vacab_size: the number of unique words
    - embedding_dim: the embedding dimension
    - n_classes: the number of final classes (tags)
    """
    super(Tashkeel, self).__init__()
    self.vocab_size = vocab_size
    # (1) Create the embedding layer
    self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)

    # it helps in initializing the first hidden layer with a size that is smaller than the input embedding.
    #This can be useful in reducing the number of parameters and providing a smooth transition from the embedding space to the hidden space.
    #  By prepending a value, you allow flexibility in the number of hidden layers in your neural network. #The layers_units list can be used to define the sizes of each hidden layer. 
    #The prepended value sets the size of the first hidden layer based on the embedding dimension.
    layers_units = [hidden_size] * 5
    layers_units = [embedding_dim // 2] + layers_units

    layers = []

    for i in range(1, len(layers_units)):
      layers.append(
          nn.LSTM(
              layers_units[i - 1] * 2,
              layers_units[i],
              bidirectional=True,
              batch_first=True,
          )
      )
    
    self.layers = nn.ModuleList(layers)
    self.projections = nn.Linear(layers_units[-1] * 2, vocab_size)
    self.layers_units = layers_units
    # self.use_batch_norm = use_batch_norm

    # # (2) Create an LSTM layer with hidden size = hidden_size and batch_first = True
    # self.lstm =  nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size,batch_first=True)

    # # (3) Create a linear layer with number of neorons = n_classes
    # self.linear =  nn.Linear(hidden_size,n_classes)
    #####################################################################################################
  
  def forward(self, sentences):
      """
      This function does the forward pass of our model
      Inputs:
      - sentences: tensor of shape (batch_size, max_length)

      Returns:
      - final_output: tensor of shape (batch_size, max_length, n_classes)
      """
      outputs = self.embedding(sentences)
      hn, cn = None, None  # Initialize hn and cn

      for i, layer in enumerate(self.layers):
          if isinstance(layer, nn.BatchNorm1d):
              continue  # Skip batch normalization layers
          if i > 0:
              outputs, (hn, cn) = layer(outputs, (hn, cn))
          else:
              outputs, (hn, cn) = layer(outputs)

      predictions = self.projections(outputs)
      output = {"diacritics": predictions}
      return output
      # final_output = None
      # embedding=self.embedding(sentences)

      # # output, (hidden_state, cell_state)
      # lstm_out,_=self.lstm(embedding)

      # linear_out=self.linear(lstm_out)

      # final_output=linear_out
      # ###############################################################################################
      # return final_output