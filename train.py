import torchvision
import torch
import torchvision.models as models
import torch.optim as opt
import torch.nn as nn
from bugDataset import bugDataset
import os

model = models.resnet50(pretrained=False)
bugData = bugDataset()
print("Dataset Size: {}".format(len(bugData)))

batch_size = 4

trainLoader = torch.utils.data.DataLoader(bugData, batch_size=batch_size, shuffle=False)
criterion = nn.CrossEntropyLoss()
optimizer = opt.SGD(model.parameters(), lr=0.1, momentum=0.09)

model.train()
model.cuda()

epoch = 4
lossRunSize = 4
runLoss = 0.0
totalIt = len(bugData)/batch_size * epoch
it = 0

for i in range(epoch):
    print("Epoch: {}".format(i))
    for j, data in enumerate(trainLoader, 0):
        image,label = data

        image = image.cuda()
        label = label.cuda()

        optimizer.zero_grad()
        output = model(image)

        loss = criterion(output, label)
        loss.backward()
        runLoss += loss.item()

        optimizer.step()

        if((it % lossRunSize)  == 0):
            run = (i + 1) * j
            runLoss /= lossRunSize
            print("[{}/{}]Average Loss: {}".format(it, totalIt, runLoss))
            runLoss = 0.0
        it += 1

saveDirectory = os.path.join("res", "model")
if not os.path.exists(saveDirectory):
    os.makedirs(saveDirectory)

savePath = os.path.join(saveDirectory, "classificationResnet50")
print("Saving model to: {}".format(savePath))
torch.save(model.state_dict(), os.path.join(savePath))
