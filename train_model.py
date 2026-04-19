import torch
from torch.utils.data import DataLoader
from torch import optim
import time
from dataset_loader import SignatureDataset
from siamese_net import SiameseNetwork, ContrastiveLoss

def train():
    # 1. Setup Data
    print("Loading dataset pairs...")
    dataset_path = "Processed_Dataset"
    
    dataset = SignatureDataset(dataset_path)
    print(f"Total pairs created: {len(dataset)}")
    
    # Simple split: 80% train, 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=32)
    
    print(f"Training pairs: {len(train_dataset)}")
    print(f"Validation pairs: {len(val_dataset)}")
    
    # 2. Setup Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    net = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    
    # 3. Training Loop
    epochs = 15 # Increased for better accuracy
    
    for epoch in range(epochs):
        start_time = time.time()
        net.train()
        train_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            img1, img2, label = data
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output1, output2 = net(img1, img2)
            
            # Calculate loss
            loss = criterion(output1, output2, label)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (i+1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        # Validation
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                img1, img2, label = data
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                out1, out2 = net(img1, img2)
                loss = criterion(out1, out2, label)
                val_loss += loss.item()
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        elapsed = time.time() - start_time
        
        print(f"--- Epoch {epoch+1} Summary ({elapsed:.1f}s) ---")
        print(f"Avg Train Loss: {avg_train_loss:.4f}")
        print(f"Avg Val Loss:   {avg_val_loss:.4f}\n")
        
    # 4. Save Model
    model_path = "signature_model.pth"
    torch.save(net.state_dict(), model_path)
    print(f"Training complete! Model saved to {model_path}")

if __name__ == "__main__":
    train()
