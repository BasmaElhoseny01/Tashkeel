from utils import *
from set_diacritics import get_diacritics_set, get_id_diacritics_dict
import pandas as pd

def result(model, test_dataset, batch_size=512):
  """
  This function takes a NER model and saves its output diacritization on a test data
  Inputs:
  - model: a NER model
  - test_dataset: dataset of type NERDataset
  """
  ########################### TODO: Replace the Nones in the following code ##########################

  # (0) Get the diacritic->id dictionary
  diacritic_id = get_id_diacritics_dict()

  # (1) create the test data loader
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  # GPU Configuration
  use_cuda = torch.cuda.is_available()
  # use_cuda=False
  device = torch.device("cuda" if use_cuda else "cpu")

  if use_cuda:
    model = model.cuda()

  counter=0
  out = {'ID':[], 'label':[]}

  # (2) disable gradients
  # block, any operations that happen won't have their gradients computed.
  with torch.no_grad():

    for test_input, test_label in tqdm(test_dataloader):
      # (3) move the test label to the device
      test_label = test_label.to(device)
      test_label=test_label.view(-1)
    
      # (4) move the test input to the device
      test_input = test_input.to(device)

      # (5) do the forward pass
      output = model(test_input)
      output=output.view(-1, output.shape[-1])

      # accuracy calculation (just add the correct predicted items to total_acc_test)
      predicted=torch.argmax(output, dim=-1)
      # print(test_label)
      predicted = predicted[test_label != 15]
      test_label = test_label[test_label != 15]

      for char in predicted:
        out['ID'].append(counter)
        out['label'].append(diacritic_id[char])
        counter += 1

  ##################################################################################################


  out = pd.DataFrame(out)
  out.to_csv('out.csv', sep=',')