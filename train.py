import torchvision
import torch
import torchvision.models as models
import torch.optim as opt
import torch.nn as nn
from bugDataset import *
import os
import math
from CAM import *


def saveModel(model, fileName):
    saveDirectory = os.path.join("res", "model")
    if not os.path.exists(saveDirectory):
        os.makedirs(saveDirectory)

    savePath = os.path.join(saveDirectory, fileName)
    print("Saving model to: {}".format(savePath))
    torch.save(model.state_dict(), os.path.join(savePath))


model = getPrepedResnet50()

imgPath = os.path.join("8-2Dataset", "train")
bugData = bugDataset(imgPath)
print("Dataset Size: {}".format(len(bugData)))

model.cuda()

batch_size = 16

trainLoader = torch.utils.data.DataLoader(bugData, batch_size=batch_size, shuffle=True)
criterion = nn.CrossEntropyLoss()
optimizer = opt.SGD(model.parameters(), lr=0.001, momentum=0.09)

model.train()

epoch = 512
lossRunSize = 50
totalIt = len(bugData)/batch_size * epoch
it = 0
epochAverageLoss = 0
accuracy = 0
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
        lossVal = loss.item()
        epochAverageLoss += lossVal

        accuracy += torch.sum((torch.argmax(output, dim=1) == label).float()).item()

        optimizer.step()

        if(j % lossRunSize  == 0):
            print("[{}/{}]Loss: {}".format(it, totalIt, lossVal))
        it += 1

    if((i + 1) % 64 == 0):
        saveModel(model, "CAM-resnet50-{}".format(i))

    print("\tAverage Loss: {}".format(epochAverageLoss/math.ceil(len(bugData)/batch_size)))
    print("\tAccuracy: {}".format(accuracy/len(bugData) * 100))
    epochAverageLoss = 0
    accuracy = 0
