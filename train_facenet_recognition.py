# train_victim_facenet.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# facenet-pytorch
from facenet_pytorch import InceptionResnetV1

#################################################################################
# We have four persons: Al, Nhat, Pierce, Yaqi
# We'll call them 0,1,2,3 in alphabetical order: ["Al","Nhat","Pierce","Yaqi"]
# Please ensure your data folder looks like:
# data/
#   train/
#     Al/     (images)
#     Nhat/
#     Pierce/
#     Yaqi/
#   val/
#     Al/
#     Nhat/
#     Pierce/
#     Yaqi/
#################################################################################

class SimpleFaceDataset(Dataset):
    """
    A straightforward dataset reading 'root_dir/class_name/*.jpg' etc.
    We'll map classes in alphabetical order: Al->0, Nhat->1, Pierce->2, Yaqi->3
    """
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        subdirs = sorted(os.listdir(root_dir))  # e.g. [Al, Nhat, Pierce, Yaqi]
        self.class_to_idx = {}
        i=0
        for sd in subdirs:
            self.class_to_idx[sd] = i
            i += 1

        for sd in self.class_to_idx.keys():
            sd_path = os.path.join(root_dir, sd)
            if not os.path.isdir(sd_path):
                continue
            label = self.class_to_idx[sd]
            for fname in os.listdir(sd_path):
                fpath = os.path.join(sd_path, fname)
                if os.path.isfile(fpath):
                    self.samples.append((fpath, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        img = Image.open(fpath).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Compose([
        T.Resize((160,160)),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    train_dir = 'data/train'
    val_dir   = 'data/val'
    train_dataset = SimpleFaceDataset(train_dir, transform=transform)
    val_dataset   = SimpleFaceDataset(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)

    num_classes = 4 

    model = InceptionResnetV1(
        classify=True,
        pretrained='vggface2', 
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epochs    = 5

    for epoch in range(epochs):
        model.train()
        total_loss=0
        total_correct=0
        total_samples=0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)  
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += imgs.size(0)

        train_acc = total_correct / total_samples
        train_loss= total_loss / total_samples

        model.eval()
        val_loss_sum = 0
        val_correct  = 0
        val_samples  = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item()*imgs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds==labels).sum().item()
                val_samples += imgs.size(0)

        val_acc  = val_correct / val_samples if val_samples>0 else 0
        val_loss = val_loss_sum / val_samples if val_samples>0 else 0

        print(f"Epoch[{epoch+1}/{epochs}] "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    # Save
    torch.save(model.state_dict(), "facenet_4class.pth")
    print("Saved facenet_4class.pth")


if __name__=="__main__":
    main()
