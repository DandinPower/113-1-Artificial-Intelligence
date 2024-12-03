import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)  # Output: 64x28x28
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 64x14x14
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # Output: 128x14x14
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 128x7x7
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)  # Output: 256x7x7

        self.fc1 = nn.Linear(256 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)  # Adds a channel dimension
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x) 
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.input_layer = nn.Linear(28 * 28, 256)
        self.hidden_layer_0 = nn.Linear(256, 256)
        self.hidden_layer_1 = nn.Linear(256, 256)
        self.hidden_layer_2 = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer_0(x))
        x = F.relu(self.hidden_layer_1(x))
        x = F.relu(self.hidden_layer_2(x))
        x = self.output_layer(x)
        return x


def train(args, model, dataloader, optimizer, epoch) -> float:
    model.train()
    total_loss = 0.0
    num_samples = len(dataloader.dataset)
    for batch_idx, (data, target) in enumerate(dataloader):
        for param in model.parameters():
            param.grad = None
        y_pred = model(data)
        loss = F.cross_entropy(y_pred, target)
        loss.backward()
        optimizer.step()
        total_loss += (loss.item() * target.shape[0])
    return total_loss / num_samples


def inference(model, dataloader) -> tuple[float, float]:
    model.eval()
    test_loss = 0
    correct = 0
    num_samples = len(dataloader.dataset)
    with torch.no_grad():
        for data, target in dataloader:
            y_pred = model(data)
            loss = F.cross_entropy(y_pred, target)
            test_loss += (loss.item() * target.shape[0])
            predicted = torch.argmax(y_pred, dim=1)
            correct += (predicted == target).sum().item()
    accuracy = correct/num_samples
    test_loss = test_loss / num_samples
    return test_loss, accuracy


def get_data_loader(device, batch_size, test_batch_size, train_valid_split):
    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}

    transform=transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_dataset = datasets.MNIST('../data', train=True, download=True)
    raw_data = train_dataset.data.float()
    transformed_data = transform(raw_data)
    targets = train_dataset.targets
    data = transformed_data.to(device)
    targets = targets.to(device)
    
    train_length = int(data.size()[0] * train_valid_split)
    train_data = data[:train_length]
    train_targets = targets[:train_length]
    valid_data = data[train_length:]
    valid_targets = targets[train_length:]
    train_dataset = torch.utils.data.TensorDataset(train_data, train_targets)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **train_kwargs)

    valid_dataset = torch.utils.data.TensorDataset(valid_data, valid_targets)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=False, **test_kwargs)

    test_dataset = datasets.MNIST('../data', train=False)
    raw_data = test_dataset.data.float()
    transformed_data = transform(raw_data)
    targets = test_dataset.targets 
    data = transformed_data.to(device)
    targets = targets.to(device)
    test_dataset = torch.utils.data.TensorDataset(data, targets)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **test_kwargs)

    return train_loader, valid_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--save_folder', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--train_valid_split', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--gamma', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    os.makedirs(args.save_folder, exist_ok=True)
    save_name = f'batch{args.batch_size}_split{args.train_valid_split}_epochs{args.epochs}_lr{args.lr}_gamma{args.gamma}'
    torch.manual_seed(args.seed)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_loader, valid_loader, test_loader = get_data_loader(device, args.batch_size, args.test_batch_size, args.train_valid_split)

    # model = Net().to(device)
    model = CNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    best_valid_loss = None
    best_valid_acc = float('-inf')
    best_test_loss = None
    best_test_acc = None
    best_epoch = None

    all_train_loss = []
    all_valid_loss = []
    all_valid_acc = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, train_loader, optimizer, epoch)
        valid_loss, valid_acc = inference(model, valid_loader)
        print(f'Epoch: {epoch}, Valid Loss: {valid_loss}, Valid Acc: {valid_acc}')
        test_loss, test_acc = inference(model, test_loader)
        print(f'Epoch: {epoch}, Test Loss: {test_loss}, Test Acc: {test_acc}')
        scheduler.step()

        if valid_acc > best_valid_acc:
            best_valid_loss = valid_loss
            best_valid_acc = valid_acc
            best_test_loss = test_loss
            best_test_acc = test_acc
            best_epoch = epoch 

        all_train_loss.append(train_loss)
        all_valid_loss.append(valid_loss)
        all_valid_acc.append(valid_acc)

    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.plot(range(args.epochs), all_train_loss, label="Training Loss", linestyle='-', color='blue')
    ax1.plot(range(args.epochs), all_valid_loss, label="Validation Loss", linestyle='--', color='orange')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss & Validation Accuracy Over Epochs")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.plot(range(args.epochs), all_valid_acc, label="Validation Accuracy", linestyle='-', color='green')
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="upper right")
    plt.savefig(f'{args.save_folder}/{save_name}.png', dpi=300)

    with open(f'{args.save_folder}/{save_name}.txt', mode='w+') as file:
        file.write(f'best_epoch,best_valid_loss,best_valid_acc,best_test_loss,best_test_acc\n')
        file.write(f'{best_epoch},{best_valid_loss},{best_valid_acc},{best_test_loss},{best_test_acc}\n')
    
if __name__ == '__main__':
    main()
