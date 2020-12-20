import torchvision
import torch
import os
from bugDataset import bugDataset

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

for i, data in enumerate(testLoader, 0):
    image, label = data

    with torch.no_grad():
        output = model(image)

    imgClass = torch.argmax(output).item()
    print(imgClass)
    if imgClass == label.item():
        correctCount += 1

print("Accuracy: {:.2f}".format(correctCount/len(testDataset) * 100))
