import random
import torch
from tqdm import tqdm
import torch.nn as nn
from model import StockPriceActionPrediction

from data_gather import StockNewsDataset, training_data
from torch.utils.data import DataLoader
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
from sklearn.preprocessing import StandardScaler
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


tokenized_news_inputs, target_array = training_data(max_length = 128)
# # Reshape and normalize
# scaler = StandardScaler()
# target_array = np.array(target_array).reshape(-1, 1)  # Shape: (N, 1)
# normalized_targets = scaler.fit_transform(target_array).squeeze()
training_dataset = StockNewsDataset(tokenized_news_inputs, target_array)
dataloader = DataLoader(training_dataset, batch_size=16, shuffle=True, num_workers=0)


model = StockPriceActionPrediction(hidden_dim=128).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss() # MSE since the outputs are continuous and we want to minimize the error in percent change

model.train() 

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0  # 👈 Accumulate loss
    progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch in progress:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].long().to(device)

        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()  # 👈 Add to epoch loss

        progress.set_postfix(batch_loss=loss.item())  # 👈 Still shows batch loss

    avg_epoch_loss = epoch_loss / len(dataloader)  # 👈 Compute average
    print(f"Epoch {epoch+1}, Final Batch Loss: {loss.item():.4f}, Avg Loss: {avg_epoch_loss:.4f}")


