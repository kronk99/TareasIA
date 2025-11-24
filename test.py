from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from sklearn.model_selection import train_test_split

class MVTecDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, 0

class MVTecDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=64, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_paths = []
        self.val_paths = []
        self.test_paths = []

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def setup(self, stage=None):
        # obtener rutas de todas las imágenes de las clases seleccionadas
        all_paths = []
        for cls in ['cable', 'capsule', 'screw', 'transistor']:
            class_dir = os.path.join(self.data_dir, cls, 'train')
            for root, _, files in os.walk(class_dir):
                all_paths += [os.path.join(root, f) for f in files if f.endswith(('.png', '.jpg'))]
        # dividir en 70/15/15
        train_paths, temp_paths = train_test_split(all_paths, test_size=0.30, random_state=42)
        val_paths, test_paths = train_test_split(temp_paths, test_size=0.50, random_state=42)
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = test_paths

    def train_dataloader(self):
        train_ds = MVTecDataset(self.train_paths, transform=self.transform)
        return DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        val_ds = MVTecDataset(self.val_paths, transform=self.transform)
        return DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        test_ds = MVTecDataset(self.test_paths, transform=self.transform)
        return DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
