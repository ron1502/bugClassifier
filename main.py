import torchvision
import torch
import torchvision.models as models
import torch.optim as opt
import torch.nn as nn
from bugDataSet import bugDataset

model = models.resnet50(pretrained=False)
bugData = bugDataset()
print("Dataset Size: {}".format(len(bugData)))
trainLoader = torch.utils.data.DataLoader(bugData, batch_size=8, shuffle=False)
criterion = nn.CrossEntropyLoss()
optimizer = opt.SGD(model.parameters(), lr=0.1, momentum=0.09)

model.train()
model.cuda()

epoch = 1
lossRunSize = 100
runLoss = 0.0

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

        if((((i +1) * j) % lossRunSize)  == 0):
            run = (i + 1) * j
            totalRun = epoch * len(dataset)
            runLoss /= lossRunSize
            print("[{}/{}]Average Loss: {}".format(j, totalRun, runLoss))
            runLoss = 0.0