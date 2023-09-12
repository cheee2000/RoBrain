from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from scipy import io


class BrainDataset(Dataset):
    def __init__(self, fmri_path, img_fea_path):
        self.fmri_data = io.loadmat(fmri_path)['data']  # img_num x trial_num x dim
        self.img_fea_data = io.loadmat(img_fea_path)['data']  # img_num x dim

    def __len__(self):
        return self.fmri_data.shape[0]

    def __getitem__(self, idx):
        return self.fmri_data[idx], self.img_fea_data[idx]

    def get_dim(self):
        return self.fmri_data.shape[-1], self.img_fea_data.shape[-1]


class ImageDataset(Dataset):
    def __init__(self, img_path, img_fea_path, imsize=64):
        self.transform = transforms.Compose([
            # transforms.Resize(int(imsize * 76 / 64)),
            # transforms.RandomCrop(imsize)
            transforms.Resize((imsize, imsize))
        ])
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [0, 1] -> [-1, 1]
        ])

        self.img_dataset = datasets.ImageFolder(img_path)
        self.img_fea_data = io.loadmat(img_fea_path)['data']

    def __getitem__(self, index):
        img = self.img_dataset[index][0]
        img = self.norm(self.transform(img))

        img_fea = self.img_fea_data[index]
        return img, img_fea

    def __len__(self):
        return len(self.img_dataset)


class BrainToImageDataset(Dataset):
    def __init__(self, img_path, fmri_path, imsize=64):
        self.transform = transforms.Compose([
            # transforms.Resize(int(imsize * 76 / 64)),
            # transforms.RandomCrop(imsize)
            transforms.Resize((imsize, imsize))
        ])
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [0, 1] -> [-1, 1]
        ])

        self.img_dataset = datasets.ImageFolder(img_path)
        self.fmri_dataset = io.loadmat(fmri_path)['data']

    def __getitem__(self, index):
        img = self.img_dataset[index][0]
        img = self.norm(self.transform(img))

        fmri_data = self.fmri_dataset[index]
        return img, fmri_data

    def __len__(self):
        return self.fmri_dataset.shape[0]

    def get_dim(self):
        return self.fmri_dataset.shape[-1]
