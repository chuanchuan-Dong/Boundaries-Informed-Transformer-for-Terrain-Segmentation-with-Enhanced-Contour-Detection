import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import torch
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToPILImage
from models import *
from loss import *
from boundary import *
class rellis_dataset(Dataset):
    """
    Using 6 classes
    """
    def __init__(self, data_root_dir, txt_file, transform=None, target_transform=None):
        """
        data_root_dit: root directory of the dataset, e.g. './data/rellis'
        txt_file: path to the train.txt listing image files
        
        """
        self.data_root_dir = data_root_dir
        self.transform = transform
        self.target_transform = target_transform
        with open(txt_file, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines()]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_root_dir, 'image', self.image_paths[index] + '.jpg')
        ann_path = os.path.join(self.data_root_dir, 'annotation', self.image_paths[index] + '_group6.png')
        image = Image.open(img_path).convert('RGB')
        annotation = Image.open(ann_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            annotation = self.target_transform(annotation)
        annotation = (annotation*256).long() # turn the label to the interger
        return image, annotation

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total_iterations = 0  
    for images, masks in tqdm(loader, desc="Training"):
        images = images.to(device)
        masks = masks.squeeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        # seg_out, boundary_out = model(images)
        # seg_loss = criterion(seg_out, masks)
        # bdy_loss = boundary_loss(boundary_out, masks)
        # loss = seg_loss + lambda_boundary * bdy_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_iterations += 1  

    
        # if total_iterations % 10 == 0:
        #     avg_loss = running_loss / (total_iterations * loader.batch_size)
        #     print(f"Iteration {total_iterations}: Training Loss: {avg_loss:.4f}")

    return running_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_iterations = 0 

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images = images.to(device)
            masks = masks.squeeze(1).to(device)

            outputs= model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)
            total_iterations += 1  

         
            # if total_iterations % 10 == 0:
            #     avg_loss = running_loss / (total_iterations * loader.batch_size)
            #     print(f"Iteration {total_iterations}: Validation Loss: {avg_loss:.4f}")

    return running_loss / len(loader.dataset)
   


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train_transforms = transforms.Compose([
#     transforms.Resize((256, 256)),  # Resize to desired size
#     transforms.ToTensor(),
# ])

# val_transforms = transforms.Compose([
#     transforms.Resize((256, 256)),  # Resize to the same size as training
#     transforms.ToTensor(),
# ])

# Data augmentation for training
train_transforms = transforms.Compose([
    transforms.Resize((256,256)),  
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

val_transforms = transforms.Compose([
    transforms.Resize((256,256)),   
    transforms.ToTensor(),   
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

target_transform = transforms.Compose([
    transforms.Resize((256,256), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor()
]) 

train_dataset = rellis_dataset(data_root_dir='./data/rellis', txt_file='./data/rellis/train.txt', transform=train_transforms, target_transform=target_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

val_dataset = rellis_dataset(data_root_dir='./data/rellis', txt_file='./data/rellis/val.txt', transform=val_transforms,target_transform=target_transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

###Create Model

model = make_SegFormerB0(num_classes=6)
# model.load_state_dict(torch.load('best_segmentation_model.pth', map_location=device)) # start from 8th iteration
model = model.to(device)

### Train parameter

class_weights_dict = {
    np.int64(0): 0.5682551429287263,
    np.int64(1): 11.452499091314577,
    np.int64(2): 0.49443918497016953,
    np.int64(3): 4.335820300418828,
    np.int64(4): 1.004562042025163,
    np.int64(5): 1.1058020928471843
}
weights_tensor = torch.tensor([class_weights_dict[np.int64(i)] for i in range(6)], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)  

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

### Start Training.

# Lists to store losses for plotting
num_epochs = 25
best_val_loss = np.inf
best_train_loss = np.inf
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = validate(model, val_loader, criterion, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    scheduler.step()

    print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Save the best model
    if val_loss < best_val_loss or train_loss < best_train_loss:
        print("Saving best model...")
        best_val_loss = val_loss
        best_train_loss = train_loss
        torch.save(model.state_dict(), f'./model_b0/best_segmentation_model_{epoch}.pth')

# Plot the training and validation losses
np.savetxt('train_losses.txt', train_losses)
np.savetxt('val_losses.txt', val_losses)

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()