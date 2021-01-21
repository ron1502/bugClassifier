import torchvision
import torch
import os
import torch.nn as nn
from bugDataset import *
from CAM import *

def test(testDataset, modelPath):
    for modelName in modelPath:
        print("Loading {}".format(modelName))
        savePath = os.path.join("res", "model", modelName)

        model = getPrepedResnet50()
        setupModel(model)
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

modelList = ["CAM-resnet50-63", "CAM-resnet50-127",
             "CAM-resnet50-191", "CAM-resnet50-255",
             "CAM-resnet50-319", "CAM-resnet50-383",
             "CAM-resnet50-447", "CAM-resnet50-511"]
imgPath = os.path.join("topLeftDatasets", "test")
testDataset = bugDataset(imgPath)

test(testDataset, modelList)
