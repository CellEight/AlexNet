import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets
from torchvision import transforms
from model import AlexNet


def train(model, train_dl, test_dl, opt, loss_func, epochs):
    train_loss = [0 for i in range(epochs)]
    test_loss = [0 for i in range(epochs)]
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss = loss_func(model(xb), yb)
            train_loss[epoch] = loss.item()
            loss.backward()
            opt.step()
            opt.zero_grad()
        with torch.no_grad():
            losses, nums = zip(*[(loss_func(model(xb),yb).item(),len(xb)) for xb, yb in test_dl])
            test_loss[epoch] = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            correct = 0
            total = 0
            for data in test_dl:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss[epoch]}, Test Loss {test_loss[epoch]}, Accuracy: {100*correct/total}')
    return train_loss, test_loss

if __name__ == "__main__":
    # Select device to train on
    device = torch.device("cuda")
    # Define transforms
    transform = transforms.Compose([torchvision.transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Load training data
    train_data = torchvision.datasets.CIFAR10(root='./data/',train=True,download=True,transform=transform)
    train_dl = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)
    test_data = torchvision.datasets.CIFAR10(root='./data/',train=False,download=True,transform=transform)
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=True, num_workers=4)
    
    epochs = 100
    model = AlexNet()
    loss_func = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train_loss, test_loss = train(model, train_dl, test_dl, opt, loss_func, epochs)
    f = open(f'model.pkl','wb')
    pickle.dump(model,f)
    f.close()


