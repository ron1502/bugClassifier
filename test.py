import torchvision
import torch
import os
from bugDataset import bugDataset
import sklearn.metrics as skm

savePath = os.path.join("res", "model", "classificationResnet50")

print("Loading Model")
model = torchvision.models.resnet50(pretrained=False)
model.load_state_dict(torch.load(savePath))

print("Loading Testing dataset")
testDataset = bugDataset("test")
testLoader = torch.utils.data.DataLoader(testDataset, batch_size=1)

model.eval()
model.cuda()

correctCount = 0
wrongCount = 0

groundTruth = []
inference = []

for i, data in enumerate(testLoader, 0):
    image, label = data

    image = image.cuda()

    with torch.no_grad():
        output = model(image)
    imgClass = torch.argmax(output).item()

    inference.append(imgClass)
    groundTruth.append(label.item())

    if imgClass == label.item():
        correctCount += 1
    else:
        wrongCount += 1

print("Accuracy: {:.2f}".format(correctCount/len(testDataset) * 100))
print("Error: {:.2f}".format(wrongCount/len(testDataset) * 100))

confMatrix = skm.confusion_matrix(groundTruth,inference)
with open("confusion_matrix.txt", "w") as file:
    file.write(str(confMatrix))
