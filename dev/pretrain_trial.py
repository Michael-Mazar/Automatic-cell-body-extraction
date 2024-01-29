from cellpose import models, io, utils
import torch
import numpy as np
from torch.utils.data import DataLoader
# from my_dataset import MyDataset # define your own dataset class

# Load the pre-trained model
model = models.CellposeModel(gpu=False, model_type='cyto')

for param in model.parameters():
    param.requires_grad = False

# from https://github.com/nigarabbasova/cellpose_segmentation/blob/main/pretrain_trial.py
"""
# Modify the last layer of the model
n_classes = 2
model.unet[-1] = torch.nn.Conv2d(64, n_classes, kernel_size=1)

# Freeze the pre-trained layers
for param in model.cp._nets[:-1].parameters():
    param.requires_grad = False

"""

"""
# Prepare the new training data
train_dataset = MyDataset('train')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.cp._nets[-1].parameters(), lr=0.001)

# Train the model
n_epochs = 10
for epoch in range(n_epochs):
    epoch_loss = 0.0
    for batch in train_loader:
        images, masks = batch
        images = images.permute(0, 3, 1, 2)
        masks = masks.permute(0, 3, 1, 2)
        outputs = model.eval(images)[-1]
        loss = criterion(outputs, masks.argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1} loss: {epoch_loss/len(train_loader):.4f}")
    
# Evaluate the model
val_dataset = MyDataset('val')
val_loader = DataLoader(val_dataset, batch_size=4)
n_correct = 0
n_total = 0
with torch.no_grad():
    for batch in val_loader:
        images, masks = batch
        images = images.permute(0, 3, 1, 2)
        masks = masks.permute(0, 3, 1, 2)
        outputs = model.eval(images)[-1]
        preds = outputs.argmax(dim=1)
        n_correct += (preds == masks.argmax(dim=1)).sum().item()
        n_total += np.prod(masks.shape)
accuracy = n_correct / n_total
print(f"Validation accuracy: {accuracy:.4f}")
"""