from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torchvision
from model import Net
import torch.optim as optim
import matplotlib.pyplot as plt

NUM_IMAGES = 859
LEARNING_RATE = .01
EPOCHS = 30
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
BATCH_SIZE = 32

class ImageDataset(Dataset):
    def __init__(self, images, labels):
        assert(len(images) == len(labels))
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def generate_samples():
    images = torch.zeros((NUM_IMAGES, 3, IMAGE_WIDTH, IMAGE_HEIGHT))
    labels = torch.zeros((NUM_IMAGES,), dtype=torch.int64)
    for i in range(NUM_IMAGES):
        path = f"images/frame{i}.jpg"
        image = torchvision.io.read_image(path)
        images[i] = image

    # read lines of labels
    with open("labels.txt") as f:
        lines = f.readlines()

    prev = 0
    for line in lines:
        line = line.strip()
        end, label = line.split(",")
        end = int(end)
        label = int(label)
        for i in range(prev, end + 1):
            if (i >= NUM_IMAGES): break
            labels[i] = label
        prev = end + 1

    arr = np.arange(NUM_IMAGES)
    np.random.shuffle(arr)
    images = images[arr]
    labels = labels[arr]

    return images, labels

images, labels = generate_samples()

split = int(NUM_IMAGES * .8)

train_dataset = ImageDataset(images=images[:split], labels=labels[:split])
test_dataset = ImageDataset(images=images[split:], labels=labels[split:])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=True)


def train(model, device, train_loader, optimizer):
    model.train()
    losses = []
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = model.loss(output, label)

        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return np.mean(losses)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss_on = model.loss(output, label, reduction='sum').item()
            test_loss += test_loss_on
            pred = output.max(1)[1]
            correct_mask = pred.eq(label.view_as(pred))
            num_correct = correct_mask.sum().item()
            correct += num_correct
        test_loss /= len(test_loader.dataset)
        test_accuracy = 100 * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

model = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


test_losses = []
test_accuracies = []
for epoch in range(1, EPOCHS + 1):
    train_loss = train(model, device, train_loader, optimizer)
    test_loss, test_accuracy = test(model, device, test_loader)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    print(epoch)
    print(f"test_loss = {test_loss}")
    print(f"Test_accuracy = {test_accuracy}")


fig, axs = plt.subplots(2)
fig.subplots_adjust(hspace=0.5)
fig.suptitle('Test Losses and Accuracies')
axs[0].plot(test_losses)
axs[0].set(xlabel='Epochs', ylabel='Loss')
axs[1].plot(test_accuracies)
axs[1].set(xlabel='Epochs', ylabel='Accuracy')
plt.show()

torch.save(model.state_dict(), "model_weights/model.pth")