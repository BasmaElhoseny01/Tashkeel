from utils import *

def train(model, train_dataset, batch_size=512, epochs=5, learning_rate=0.01):
    # (1) create the dataloader of the training set (make the shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # (2) make the criterion cross-entropy loss
    criterion = CrossEntropyLoss()

    # (3) create the optimizer (Adam)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    # (4) Check for GPU availability
    use_cuda = torch.cuda.is_available()

    # (5) Move model and criterion to GPU if available
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    criterion.to(device)

    print('GPU Availability ? -->', use_cuda)

    if use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            # (6) move the train input and label to the device
            train_label = train_label.to(device)
            train_input = train_input.to(device)

            # (7) do the forward pass
            output = model(train_input)

            # (8) reshape output and calculate the loss
            output = output["diacritics"].view(-1, output["diacritics"].shape[-1])
            train_label = train_label.view(-1)
            batch_loss = criterion(output, train_label)

            # (9) append the batch loss to the total_loss_train
            total_loss_train += batch_loss.item()

            # (10) calculate the batch accuracy
            acc = (torch.argmax(output, dim=-1) == train_label).sum().item()
            total_acc_train += acc

            # (11) zero your gradients
            optimizer.zero_grad()

            # (12) do the backward pass
            batch_loss.backward()

            # (13) update the weights with your optimizer
            optimizer.step()

        # epoch loss
        epoch_loss = total_loss_train / len(train_dataset)

        # (14) calculate the accuracy
        epoch_acc = total_acc_train / (len(train_dataset) * (train_dataset[0][0].shape[0]))

        print(f'Epochs: {epoch_num + 1} | Train Loss: {epoch_loss} | Train Accuracy: {epoch_acc}\n')

    ##############################################################################################################
