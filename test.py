import torchvision
import torch
import os
from bugDataset import bugDataset


def test(testDataset, modelPath):
    for modelName in modelPath:
        print("Loading {}".format(modelName))
        savePath = os.path.join("res", "model", modelName)

        model = torchvision.models.resnet50(pretrained=False)
        model.load_state_dict(torch.load(savePath))

        print("Loading Testing dataset")
        testLoader = torch.utils.data.DataLoader(testDataset, batch_size=1)

        model.eval()
        model.cuda()

        correctCount = 0


        for i, data in enumerate(testLoader, 0):
            image, label = data

            image = image.cuda()

            with torch.no_grad():
                output = model(image)

            imgClass = torch.argmax(output).item()


            if imgClass == label.item():
                correctCount += 1

        print("Accuracy: {:.2f}".format(correctCount/len(testDataset) * 100))

modelList = ["classificationResnet50-191"]
testDataset = bugDataset("test")

test(testDataset, modelList)
