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
    "105ALBOS": "Anthrax albosparsus (Bigot)",
    "106MACQU": "Anthrax albofasciatus Macquart",
    "107TUCKE": "Anastoechus melanohalteralis Tucker",
    "119SACKE": "Anastoechus barbatus Osten Sacken",
    "123COQUI": "Aldrichia ehrmanii Coquillett",
    "108ANTSP": "Anthrax sp.",
    "141ANPWI": "Anthrax pluto Wiedemann",
    "110ANPLO": "Anthrax pauper (Loew)",
    "111ANLSA": "Anthrax limatulus Say",
    "112ANLMA": "Anhrax laticellus Marston",
    "113ANISA": "Anthrax irroratus Say",
    "114ANAWA": "Anthrax antecedens Walker",
    "115APOSA": "Aphoebantas conurus Osten Sacken",
    "116APMSA": "Aphoebantas mus (Osten Sacken)",
    "136APHSP": "Aphoebantus sp.",
    "117ASAOS": "Astrophanes adonis Osten Sacken",
    "118BOALO": "Bombylius albicapillus Loew",
    "137BOATL": "Bombylius atriceps Loew",
    "138BOCIM": "Bombylius cinerascens Mikan",
    "139BODIM": "Bombylius discolor Mikan",
    "140BOLAS": "Bombylius lancifer Osten Sacken",
    "120BOMLI": "Bombylius major Linnaeus",
    "121BOMPA": "Bombylius medorae Pointer",
    #"122BOOSA": "Bombylius metopium Osten Sacken", No specimens
    "124BOMWI": "Bombylius mexicanus Wiedemann",
    "125BOPLO": "Bombylius pulchellus Loew",
    "126BOVFA": "Bombylius varius Fabricius",
    "127BOCOQ": "Bombylius (Parambombylius) coqilletti (Williston)",
    "128CHCME": "Chrysanthrax cypris (Meigen)",
    "128CHESA": "Chrysanthrax edititius (Say)",
    "129CHECU": "Chrysanthrax eudorus (Coquillett)",
    "129DISOS": "Dipalta serpentina Osten Sacken",
    "130DOGRA": "Dolichomya gracilis Wiedemann",
    "131ECMOS": "Eclimus mangnus (Osten Sacken)",
    "142GEALP": "Geron albarius Painter",
    "132GEAPA": "Geron arenicola Painter",
    "135GEDCR": "Geron digitarius Cresson",
    "134GEPAI": "Geron Parvidus Painter",
    "143GESPA": "Geron (Empidigeron) snowi Painter",
    "144GEVLO": "Geron vitripennis Leow",
    "145GERSP": "Geron sp.",
    "146HEOSA": "Hemipenthes eumenes Osten Sacken",
    "147HEICO": "Hemipenthes inops (Coquillett)",
    "148HEISA": "Hemipenthes lepidota Osten Sacken"
}

indexToLabel = ["Anthrax aterrimus (Bigot)",
                "Anthrax argyropyga Wiedemann",
                "Anthrax analis Say",
                "Anthrax albosparsus (Bigot)",
                "Anthrax albofasciatus Macquart",
                "Anastoechus melanohalteralis Tucker",
                "Anastoechus barbatus Osten Sacken",
                "Aldrichia ehrmanii Coquillett",
                "Anthrax sp.",
                "Anthrax pluto Wiedemann",
                "Anthrax pauper (Loew)",
                "Anthrax limatulus Say",
                "Anhrax laticellus Marston",
                "Anthrax irroratus Say",
                "Anthrax antecedens Walker",
                "Aphoebantas conurus Osten Sacken",
                "Aphoebantas mus (Osten Sacken)",
                "Aphoebantus sp.",
                "Astrophanes adonis Osten Sacken",
                "Bombylius albicapillus Loew",
                "Bombylius atriceps Loew",
                "Bombylius cinerascens Mikan",
                "Bombylius discolor Mikan",
                "Bombylius lancifer Osten Sacken",
                "Bombylius major Linnaeus",
                "Bombylius medorae Pointer",
                #"Bombylius metopium Osten Sacken", No specimens
                "Bombylius mexicanus Wiedemann",
                "Bombylius pulchellus Loew",
                "Bombylius varius Fabricius",
                "Bombylius (Parambombylius) coqilletti (Williston)",
                "Chrysanthrax cypris (Meigen)",
                "Chrysanthrax edititius (Say)",
                "Chrysanthrax eudorus (Coquillett)",
                "Dipalta serpentina Osten Sacken",
                "Dolichomya gracilis Wiedemann",
                "Eclimus mangnus (Osten Sacken)",
                "Geron albarius Painter",
                "Geron arenicola Painter",
                "Geron digitarius Cresson",
                "Geron Parvidus Painter",
                "Geron (Empidigeron) snowi Painter",
                "Geron vitripennis Leow",
                "Geron sp.",
                "Hemipenthes eumenes Osten Sacken",
                "Hemipenthes inops (Coquillett)",
                "Hemipenthes lepidota Osten Sacken"]

class bugDataset(Dataset):
    def  __init__(self, sourcePath="imgs"):
        super(bugDataset, self).__init__()
        absPath = os.path.abspath(sourcePath)
        bugFolders = os.listdir(absPath)
        prepImg = trans.Compose([
            trans.ToTensor(),
            trans.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.images = []
        self.labels = []

        for folder in bugFolders:
            path = os.path.join(absPath, folder)
            imgList = os.listdir(path)
            for i in tqdm(range(0, len(imgList)), folderToName[folder]):
                imgName = imgList[i]
                img = Image.open(os.path.join(path,imgName)).resize((376 , 250))
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
