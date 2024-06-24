import torch
import torch.optim as optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset
from vision_mamba.model import Vim
import numpy as np
import time

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a random tensor for inputs and targets
inputs = torch.randn(100, 3, 286, 286)  # 100 images
targets = torch.randn(100, 1)  # 100 target values

# Create a TensorDataset and DataLoader
dataset = TensorDataset(inputs, targets)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Initialize the Vim model
model = Vim(
    dim=256,
    # heads=8,
    dt_rank=32,
    dim_inner=256,
    d_state=256,
    num_classes=1,  # For regression, typically the output is a single value per instance
    image_size=286,
    patch_size=13,
    channels=3,
    dropout=0.1,
    depth=12,
)

# Move the model to the GPU
model.to(device)

# Using Mean Squared Error Loss for a regression task
criterion = MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()  # Set the model to training mode
num_epochs = 5  # Define the number of epochs
verbose = True  # Set verbose to True to print correlation

# Record the start time
start_time = time.time()

for epoch in range(num_epochs):
    total_loss = 0.0
    num_batches = 0
    outputs_all = []
    targets_all = []

    for batch_inputs, batch_targets in train_loader:
        # Move the inputs and targets to the GPU
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        optimizer.zero_grad()  # Zero the parameter gradients

        # Forward pass
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1

        # Debugging shapes
        print("Output shape:", outputs.shape)
        print("Target shape:", batch_targets.shape)

        # Collect outputs and targets for correlation, ensure they are flattened
        outputs_all.append(outputs.view(-1).detach().cpu().numpy())
        targets_all.append(batch_targets.view(-1).detach().cpu().numpy())

    # Calculate average loss for the epoch
    average_loss = total_loss / num_batches
    print(f'Epoch {epoch + 1}: Average Loss {average_loss:.4f}')

    # Compute correlation
    outputs_flat = np.concatenate(outputs_all)
    targets_flat = np.concatenate(targets_all)
    corr = np.corrcoef(outputs_flat, targets_flat)[0, 1]
    if verbose:
        print('Epoch {}: Correlation: {:.4f}'.format(epoch + 1, corr))

# Record the end time
end_time = time.time()

# Calculate and print the total training time
total_training_time = end_time - start_time
print(f'Total Training Time: {total_training_time:.2f} seconds')
