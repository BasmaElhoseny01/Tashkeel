from utils import *
from Globals import vocab_map

# def evaluate(model, test_dataset, batch_size=512):
def evaluate(model, test_dataset, batch_size=2):
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
  counter=0

  # (2) disable gradients
  # block, any operations that happen won't have their gradients computed.
  with torch.no_grad():

    for test_input, test_label in tqdm(test_dataloader):
      # print("....................")
      # print(test_label)

      # (3) move the test input to the device
      test_label = test_label.to(device)
      test_label=test_label.view(-1)

      # Ignore List
      ignore_list=set([vocab_map['<s>'],vocab_map['</s>'],vocab_map[' '],vocab_map['<PAD>'],vocab_map['<UNK>']])
      # print("ignore_list",ignore_list)
      input_mask=np.array([[value.item() not in ignore_list for value in row] for row in test_input])
      input_mask=input_mask.reshape((input_mask.shape[0]*input_mask.shape[1]))

    
      # (4) move the test label to the device
      test_input = test_input.to(device)

      # (5) do the forward pass
      output = model(test_input)
      output=output.view(-1, output.shape[-1])

      # accuracy calculation (just add the correct predicted items to total_acc_test)
      predicted=torch.argmax(output, dim=-1)
      # print(predicted)


      # print(test_label)
      # predicted = predicted[test_label != 15]
      predicted=predicted[input_mask]
      test_label = test_label[input_mask]
      print(">..........")
      print(test_label)
      print(predicted)

      acc = (predicted == test_label).sum().item()
      # print("acc1",acc/test_label.size(0))
      # print("acc2",total_acc_test/test_label.size(0))

      total_acc_test += acc
      counter+=test_label.size(0)
      break
      

      # print("counter",test_label.size(0),counter)
      # print("acc3",total_acc_test,acc)
      # print("total_acc_test",total_acc_test/counter)

    # (6) calculate the over all accuracy
    total_acc_test /= counter
  ##################################################################################################


  print(f'\nTest Accuracy: {total_acc_test}')