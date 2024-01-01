from utils import *
# from set_diacritics import get_diacritics_set, get_id_diacritics_dict
import pandas as pd
import sys

from Globals import vocab_map
# def result(model, test_dataset, batch_size=512):
def result(model, test_dataset, batch_size=2):
  """
  This function takes a NER model and saves its output diacritization on a test data
  Inputs:
  - model: a NER model
  - test_dataset: dataset of type NERDataset
  """
  ########################### TODO: Replace the Nones in the following code ##########################

  # (0) Get the diacritic->id dictionary

  # (1) create the test data loader
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  # GPU Configuration
  use_cuda = torch.cuda.is_available()
  # use_cuda=False
  device = torch.device("cuda" if use_cuda else "cpu")

  if use_cuda:
    model = model.cuda()
    
  csv_dictionary = {'ID':[], 'label':[]}
  csv_counter=0

  # (2) disable gradients
  # block, any operations that happen won't have their gradients computed.
  with torch.no_grad():

    for test_input in tqdm(test_dataloader):

      # Ignore List
      ignore_list=set([vocab_map['<s>'],vocab_map['</s>'],vocab_map[' '],vocab_map['<PAD>'],vocab_map['<UNK>']])
      # print("ignore_list",ignore_list)
      input_mask=np.array([[value.item() not in ignore_list for value in row] for row in test_input])
      input_mask=input_mask.reshape((input_mask.shape[0]*input_mask.shape[1]))

      # (4) move the test input to the device
      test_input = test_input.to(device)
      # No of senetces in batch * Tmax

      # (5) do the forward pass
      output = model(test_input)
      output=output.view(-1, output.shape[-1])

      # accuracy calculation (just add the correct predicted items to total_acc_test)
      predicted=torch.argmax(output, dim=-1)
      predicted=predicted[input_mask]
      print(predicted)




      for char in predicted:
        csv_dictionary['ID'].append(csv_counter)
        csv_dictionary['label'].append(char.cpu().item())
        csv_counter += 1


  # ##################################################################################################
  csv_dictionary = pd.DataFrame(csv_dictionary)
  print(csv_dictionary)
  csv_dictionary.to_csv('csv_dictionary.csv', sep=',',index=False)