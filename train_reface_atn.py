import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from tqdm import tqdm
import itertools

# U-Net Attack Generator
class UNetAttackGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetAttackGenerator, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Output layer
        self.outconv = nn.Conv2d(64, out_channels, 1)
        self.tanh = nn.Tanh()  # To bound the perturbation

    def forward(self, x):
        # Encoding
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        
        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(pool3)
        
        # Decoding
        upconv3 = self.upconv3(bottleneck)
        concat3 = torch.cat((upconv3, enc3), dim=1)
        dec3 = self.dec3(concat3)
        
        upconv2 = self.upconv2(dec3)
        concat2 = torch.cat((upconv2, enc2), dim=1)
        dec2 = self.dec2(concat2)
        
        upconv1 = self.upconv1(dec2)
        concat1 = torch.cat((upconv1, enc1), dim=1)
        dec1 = self.dec1(concat1)
        
        # Output
        perturbation = self.outconv(dec1)
        perturbation = self.tanh(perturbation) * 0.05  # Scale perturbation to be small
        
        # Return the perturbation, not the adversarial image
        return perturbation

class FaceAttackDataset(Dataset):
    def __init__(self, root_dir, source_idx, target_idx, transform=None):
        """
        Args:
            root_dir (string): Directory with images.
            source_idx (int): Source class index to attack.
            target_idx (int): Target class index to fool the model into predicting.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.source_idx = source_idx
        self.target_idx = target_idx
        self.samples = []
        self.class_to_idx = {}
        
        # Get directory list
        subdirs = sorted(os.listdir(root_dir))
        
        # Create class to index mapping
        for i, sd in enumerate(subdirs):
            self.class_to_idx[sd] = i
        
        # Get only samples from source class
        for sd in self.class_to_idx.keys():
            label = self.class_to_idx[sd]
            if label != source_idx:
                continue  # Only include source class
            
            sd_path = os.path.join(root_dir, sd)
            if not os.path.isdir(sd_path):
                continue
                
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
        return img, label, self.target_idx

def train_attack(source_person, target_person, epsilon=0.05, epochs=5):
    """
    Train an attack model to make source_person be recognized as target_person
    
    Args:
        source_person (str): Name of the person to attack
        target_person (str): Name of the person to impersonate
        epsilon (float): Maximum perturbation magnitude
        epochs (int): Number of training epochs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data transforms - same as your face recognition model
    transform = T.Compose([
        T.Resize((160, 160)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Load class mapping
    class_labels = ["Al", "Nhat", "Pierce", "Yaqi"]
    source_idx = class_labels.index(source_person)
    target_idx = class_labels.index(target_person)
    
    print(f"Training attack: {source_person}({source_idx}) → {target_person}({target_idx})")
    
    # Create dataset focused on source person
    train_dir = 'data/train'
    
    # Create a dataset with source person images
    dataset = FaceAttackDataset(
        root_dir=train_dir, 
        source_idx=source_idx,
        target_idx=target_idx,
        transform=transform
    )
    
    if len(dataset) == 0:
        print(f"Error: No samples found for person {source_person}")
        return False
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Load the victim face recognition model
    face_model = InceptionResnetV1(
        classify=True,
        pretrained=None,
        num_classes=len(class_labels)
    ).to(device)
    
    face_model.load_state_dict(torch.load("facenet_4class.pth", map_location=device))
    face_model.eval()  # Set to evaluation mode
    
    # Freeze the face recognition model
    for param in face_model.parameters():
        param.requires_grad = False
    
    # Create the attack generator model
    attack_model = UNetAttackGenerator().to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        attack_success = 0
        total_samples = 0
        
        attack_model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, orig_labels, target_labels in progress_bar:
            images = images.to(device)
            orig_labels = orig_labels.to(device)
            target_labels = target_labels.to(device)
            
            # Generate perturbation
            perturbation = attack_model(images)
            
            # Create adversarial images
            adv_images = torch.clamp(images + perturbation, -1, 1)
            
            # Get predictions from face recognition model
            with torch.no_grad():
                clean_outputs = face_model(images)
            
            adv_outputs = face_model(adv_images)
            target_loss = criterion(adv_outputs, target_labels)
            l2_loss = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), dim=1))
            loss = target_loss + 0.1 * l2_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            clean_preds = torch.argmax(clean_outputs, dim=1)
            adv_preds = torch.argmax(adv_outputs, dim=1)
            
            success = (adv_preds == target_labels).float().sum().item()
            attack_success += success
            total_samples += images.size(0)
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'attack_success': f"{success}/{images.size(0)}"
            })
        
        avg_loss = total_loss / len(dataloader)
        success_rate = attack_success / total_samples * 100
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
              f"Attack Success Rate: {success_rate:.2f}%")
    
    # Save the trained attack model
    model_save_path = f"attack_{source_person}_to_{target_person}.pth"
    torch.save(attack_model.state_dict(), model_save_path)
    print(f"Attack model saved to {model_save_path}")
    
    return True

def train_all_attacks(epochs=5):
   
    class_labels = ["Al", "Nhat", "Pierce", "Yaqi"]
    attack_pairs = [(src, tgt) for src in class_labels for tgt in class_labels if src != tgt]
    
    print(f"Will train {len(attack_pairs)} combinations")
    
    successful_models = 0
    
    for source_person, target_person in attack_pairs:
        print(f"\n{'='*60}")
        print(f"Training attack: {source_person} → {target_person}")
        print(f"{'='*60}")
        
        success = train_attack(source_person, target_person, epochs=epochs)
        if success:
            successful_models += 1
    
    print(f"\nTraining success! Succeed trained {successful_models}/{len(attack_pairs)} attack models.")

if __name__ == "__main__":
    train_all_attacks(epochs=10)