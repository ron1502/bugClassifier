import torchvision
import torch
import os
import torch.nn as nn
from bugDataset import *


def test(testDataset, modelPath):
    for modelName in modelPath:
        print("Loading {}".format(modelName))
        savePath = os.path.join("res", "model", modelName)

        model = torchvision.models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, len(indexToLabel), True)
        model.load_state_dict(torch.load(savePath))

        print("Loading Testing dataset")
        testLoader = torch.utils.data.DataLoader(testDataset, batch_size=1)

        model.eval()
        model.cuda()

        correctCount = 0

        wrongCount = {}

        for i, data in enumerate(testLoader, 0):
            image, label = data

            image = image.cuda()

            with torch.no_grad():
                output = model(image)

            imgClass = torch.argmax(output).item()


            if imgClass == label.item():
                correctCount += 1
            else:
                gTruth = label.item()
                if gTruth not in wrongCount:
                    wrongCount[gTruth] = 0
                wrongCount[gTruth] += 1

        print("Accuracy: {:.2f}".format(correctCount/len(testDataset) * 100))
        '''
        with open("{}.txt".format(modelName), "w") as f:
            for key in wrongCount:
                f.write("{}: {}\n".format(indexToLabel[key], wrongCount[key]))
        '''

modelList = ["resnet50-63", "resnet50-127",
             "resnet50-191", "resnet50-255",
             "resnet50-319", "resnet50-383",
             "resnet50-447", "resnet50-511"]
imgPath = os.path.join("topLeftDatasets", "test")
testDataset = bugDataset(imgPath)

test(testDataset, modelList)
