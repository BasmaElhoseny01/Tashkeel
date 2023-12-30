from utils import *



def train(model, train_dataset, batch_size=512, epochs=5, learning_rate=0.01):
  """
  This function implements the training logic
  Inputs:
  - model: the model ot be trained
  - train_dataset: the training set of type NERDataset
  - batch_size: integer represents the number of examples per step
  - epochs: integer represents the total number of epochs (full training pass)
  - learning_rate: the learning rate to be used by the optimizer
  """

  ############################## TODO: replace the Nones in the following code ##################################

  # (1) create the dataloader of the training set (make the shuffle=True)
  # data loader will randomly shuffle the dataset before each epoch. Shuffling is an important practice during training,
  #  and it helps ensure that the model does not learn patterns based on the order of the data.
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)

  # (2) make the criterion cross entropy loss
  # Assuming a multi-class classification task with C classes
  criterion = CrossEntropyLoss()

  # (3) create the optimizer (Adam)
  optimizer = torch.optim.Adam(params=model.parameters(
    ), lr=learning_rate)  # fill this with correct code

  # GPU configuration
  use_cuda = torch.cuda.is_available()
  # use_cuda = False

  device = torch.device("cuda" if use_cuda else "cpu")
  # device = torch.device("cpu")
  if use_cuda:
    print("Using GPU")
    model = model.cuda()
    criterion = criterion.cuda()
  else: print("Using CPU")

  for epoch_num in range(epochs):
    total_acc_train = 0
    total_loss_train = 0

    for train_input, train_label in tqdm(train_dataloader):


      # (4) move the train input to the device
      # Size (batch_size,T)
      train_label = train_label.to(device)

      # (5) move the train label to the device
      # Size (batch_size,T)
      train_input = train_input.to(device)


      # (6) do the forward pass
      # Size (batch_size,T)
      output = model(train_input)
      # print("train_input",train_input.size())
      # print("train_label",train_label.size())
      # print("output",output.size())

      # (7) loss calculation (you need to think in this part how to calculate the loss correctly)
      # Reshape output to (batch_size * T, num_classes)
      output=output.view(-1, output.shape[-1])
      # Reshape ground truth labels to (batch_size * T)
      train_label=train_label.view(-1)
      # print("train_label",train_label.size())
      # print(train_label)
      # print(train_label.dtype)
      # print("output",output.size())
      # print(output)
      # print(output.dtype)
      batch_loss = criterion(output,train_label)

      # (8) append the batch loss to the total_loss_train
      total_loss_train += batch_loss

      # (9) calculate the batch accuracy (just add the number of correct predictions)
      # print(torch.argmax(output, dim=-1))
      # print(torch.argmax(output, dim=-1) == train_label)
      acc = (torch.argmax(output, dim=-1) == train_label).sum().item()
      total_acc_train += acc

      # (10) zero your gradients
      optimizer.zero_grad()

      # (11) do the backward pass
      batch_loss.backward()

      # (12) update the weights with your optimizer
      optimizer.step()

    # epoch loss
    # average loss per sample for the entire training
    epoch_loss = total_loss_train / len(train_dataset)

    # (13) calculate the accuracy
    epoch_acc = total_acc_train / (len(train_dataset)* (train_dataset[0][0].shape[0]))   #[.. *104(T)]

    print(
        f'Epochs: {epoch_num + 1} | Train Loss: {epoch_loss} \
        | Train Accuracy: {epoch_acc}\n')

  ##############################################################################################################