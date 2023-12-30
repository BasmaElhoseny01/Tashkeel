from utils import *

def evaluate(model, test_dataset, batch_size=512):
  """
  This function takes a NER model and evaluates its performance (accuracy) on a test data
  Inputs:
  - model: a NER model
  - test_dataset: dataset of type NERDataset
  """
  ########################### TODO: Replace the Nones in the following code ##########################

  # (1) create the test data loader
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False)

  # GPU Configuration
  use_cuda = torch.cuda.is_available()
  # use_cuda=False
  device = torch.device("cuda" if use_cuda else "cpu")

  if use_cuda:
    model = model.cuda()

  total_acc_test = 0

  # (2) disable gradients
  # block, any operations that happen won't have their gradients computed.
  with torch.no_grad():

    for test_input, test_label in tqdm(test_dataloader):
      # (3) move the test input to the device
      test_label = test_label.to(device)

      # (4) move the test label to the device
      test_input = test_input.to(device)

      # (5) do the forward pass
      output = output = model(test_input)

      # accuracy calculation (just add the correct predicted items to total_acc_test)
      acc = acc = (torch.argmax(output, dim=-1) == test_label).sum().item()
      total_acc_test += acc

    # (6) calculate the over all accuracy
    total_acc_test /= (len(test_dataset) * test_dataset[0][0].shape[0])
  ##################################################################################################


  print(f'\nTest Accuracy: {total_acc_test}')