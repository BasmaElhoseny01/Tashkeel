from utils import *
from Globals import vocab_map,classes

class TashkeelDatasetTest(torch.utils.data.Dataset):
  def __init__(self, x, pad,max_length):
    """
    This is the constructor of the NERDataset
    Inputs:
    - x: a list of lists where each list contains the ids of the tokens
    - y: a list of lists where each list contains the label of each token in the sentence
    - pad: the id of the <PAD> token (to be used for padding all sentences and labels to have the same length)
    - max_length: max_length of sequence(T)
    """
    x_padded=[]

    print("Padding ..... with ",max_length,":",len(x))

    # Add <s> </s> then split
    for i in range(len(x)):

      setence=x[i]

      # Add <s> and </s>
      setence.insert(0, vocab_map['<s>'])

      setence.append(vocab_map['</s>'])

      # if senetce is too long split into 2 setences
      if(len(setence)>max_length):

        # Split Characters
        splitted_x = [list(group) for group in zip_longest(*[iter(setence)] * max_length, fillvalue=pad)]

        x_padded.extend(splitted_x)

    
      elif(len(setence)<max_length):
        # print("<",len(setence))
        # To chars
        padded_list_x = setence + [pad] * (max_length - len(setence))

        x_padded.extend([padded_list_x])
      
      else:
        # Equal Length
        x_padded.extend([setence])

    print("Done Padding .....", len(x_padded))
    for i,sen in enumerate(x_padded):
      # print(len(x_padded[i]))
      # print(len(y_padded[i]))
      # print("...........")
      if( len(x_padded[i])!=max_length ):
        print("Errrrrrrrrr",i,len(x_padded[i]))
    

    # Add Features Extractions
    


    



    self.x = torch.tensor(x_padded)


  def __len__(self):
    """
    This function should return the length of the dataset (the number of sentences)
    """
    return len(self.x)
  
  def __getitem__(self, idx):
    """
    This function returns a subset of the whole dataset
    """
    return (self.x[idx])
    
