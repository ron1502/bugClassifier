from torch.utils.data import Dataset
import torchvision.transforms as trans
import torch
import PIL.Image as Image
import os
from tqdm import tqdm


folderToName = {
    "102ATERR": "Anthrax aterrimus (Bigot)",
    "103WIEDE": "Anthrax argyropyga Wiedemann",
    "104ANASA": "Anthrax analis Say",
    "105ALBOS": "Anthrax albosparsus (Bigot)"
}

indexToLabel = ["Anthrax aterrimus (Bigot)",
                "Anthrax argyropyga Wiedemann",
                "Anthrax analis Say",
                "Anthrax albosparsus (Bigot)"]

class bugDataset(Dataset):
    def  __init__(self, sourcePath="imgs"):
        super(bugDataset, self).__init__()
        absPath = os.path.abspath(sourcePath)
        bugFolders = os.listdir(absPath)
        prepImg = trans.Compose([
            trans.ToTensor(),
            trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.images = []
        self.labels = []
        for folder in bugFolders:
            path = os.path.join(absPath, folder)
            imgList = os.listdir(path)
            for i in tqdm(range(0, len(imgList)), folderToName[folder]):
                imgName = imgList[i]
                img = Image.open(os.path.join(path,imgName)).resize((376, 250))
                img = prepImg(img)
                self.images.append(img)
                self.labels.append(self.getImgLabel(folder))


    def getImgLabel(self, folderName):
        return indexToLabel.index(folderToName[folderName])

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    dataset = bugDataset()
    print("Total Size: {}".format(len(dataset)))
