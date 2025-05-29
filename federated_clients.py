"""
FIXED Federated Learning Client Simulator
Uses YOUR proven preprocessing from KNN analysis
CLEAN version with consistent naming
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import copy
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class AlzheimerDataset(Dataset):
    """Custom dataset for Alzheimer's data"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ImprovedAlzheimerNet(nn.Module):
    """Improved Neural Network based on your successful preprocessing"""
    
    def __init__(self, input_size: int = 7, hidden_size: int = 64, dropout_rate: float = 0.2):
        super(ImprovedAlzheimerNet, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),  # Removed BatchNorm to avoid single-sample issues
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),  # Removed BatchNorm to avoid single-sample issues
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate // 2),
            
            nn.Linear(hidden_size // 4, 2)  # Binary classification
        )
    
    def forward(self, x):
        return self.network(x)

class ImprovedFederatedClient:
    """FIXED client using your proven preprocessing approach"""
    
    def __init__(self, client_id: str, data: pd.DataFrame, model: nn.Module):
        self.client_id = client_id
        self.data = data
        self.model = copy.deepcopy(model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        
        # IMPORTANT: Initialize scaler attribute first
        self.scaler = None
        self.feature_names = []
        
        # Prepare local dataset using YOUR preprocessing approach
        self.train_loader, self.test_loader = self._prepare_data_your_way()
        
        print(f"🏥 Client {self.client_id} initialized with {len(self.data)} samples")
    
    def _prepare_data_your_way(self) -> Tuple[DataLoader, DataLoader]:
        """Use YOUR proven preprocessing approach from KNN analysis"""
        
        print(f"  📊 {self.client_id} - Applying YOUR proven preprocessing...")
        
        # Step 1: Handle missing values (YOUR approach - complete removal)
        data_clean = self.data.dropna().copy()
        print(f"    After removing NaN: {len(data_clean)} samples")
        
        # Step 2: Remove "Converted" class (YOUR approach - binary classification)
        data_processed = data_clean[data_clean["Group"] != "Converted"].copy()
        print(f"    After removing 'Converted': {len(data_processed)} samples")
        
        if len(data_processed) == 0:
            print(f"    ⚠️  No data left after preprocessing, using fallback")
            data_processed = data_clean.copy()
        
        # Step 3: Encode categorical variables (YOUR approach)
        group_to_idx = {'Nondemented': 0, 'Demented': 1}
        sex_to_idx = {'M': 0, 'F': 1}
        
        data_processed = data_processed.replace(group_to_idx)
        data_processed = data_processed.replace(sex_to_idx)
        
        # Step 4: Feature selection (YOUR approach - remove CDR and MMSE)
        available_features = ['Age', 'EDUC', 'SES', 'eTIV', 'nWBV', 'ASF', 'M/F']
        feature_cols = [col for col in available_features if col in data_processed.columns]
        
        print(f"    Selected features: {feature_cols}")
        
        X = data_processed[feature_cols].values
        y = data_processed['Group'].values
        
        # Ensure binary classification
        y = np.where(y == 'Converted', 1, y)
        y = y.astype(int)
        
        print(f"    Class distribution: {np.bincount(y)}")
        
        # Step 5: Standard scaling (YOUR approach)
        self.scaler = StandardScaler()  # FIXED: Set scaler attribute
        X_scaled = self.scaler.fit_transform(X)
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Step 6: Train/test split
        if len(X_scaled) > 4:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.3, random_state=42, 
                    stratify=y if len(np.unique(y)) > 1 else None
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.3, random_state=42
                )
        else:
            X_train, X_test = X_scaled, X_scaled
            y_train, y_test = y, y
        
        # Create datasets
        train_dataset = AlzheimerDataset(X_train, y_train)
        test_dataset = AlzheimerDataset(X_test, y_test)
        
        # Create data loaders
        batch_size = min(8, max(1, len(train_dataset) // 2))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=min(8, len(test_dataset)), shuffle=False)
        
        print(f"    Train/Test split: {len(train_dataset)}/{len(test_dataset)}")
        
        return train_loader, test_loader
    
    def local_train(self, epochs: int = 10) -> Dict[str, torch.Tensor]:
        """Enhanced training"""
        print(f"  🔄 Client {self.client_id} starting enhanced training...")
        
        self.model.train()
        initial_params = copy.deepcopy(dict(self.model.named_parameters()))
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            if epoch % max(1, epochs // 3) == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Calculate parameter updates
        gradients = {}
        current_params = dict(self.model.named_parameters())
        
        for name in initial_params:
            if initial_params[name].requires_grad:
                gradients[name] = initial_params[name].data - current_params[name].data
        
        print(f"  ✅ Client {self.client_id} enhanced training complete")
        return gradients
    
    def evaluate_model(self) -> Dict[str, float]:
        """Enhanced evaluation"""
        self.model.eval()
        
        correct = 0
        total = 0
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                loss = self.criterion(output, target)
                
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                total_loss += loss.item()
                num_batches += 1
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }
    
    def update_model(self, global_gradients: Dict[str, torch.Tensor]):
        """Update local model"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in global_gradients and param.requires_grad:
                    param.data -= global_gradients[name]
        
        print(f"  📥 Client {self.client_id} model updated")

class ImprovedClientManager:
    """FIXED client manager"""
    
    def __init__(self, csv_file_path: str, num_clients: int = 3):
        self.csv_file_path = csv_file_path
        self.num_clients = num_clients
        self.clients = []
        
        # Load and distribute data
        self._load_and_distribute_data_your_way()
    
    def _load_and_distribute_data_your_way(self):
        """Load using YOUR proven approach"""
        print("📊 Loading dataset with YOUR proven preprocessing approach...")
        
        df = pd.read_csv(self.csv_file_path)
        print(f"Raw data: {len(df)} samples")
        print(f"Original classes: {df['Group'].value_counts().to_dict()}")
        
        # YOUR preprocessing
        df_clean = df.dropna()
        print(f"After removing NaN: {len(df_clean)} samples")
        
        df_processed = df_clean[df_clean["Group"] != "Converted"].copy()
        print(f"After removing 'Converted': {len(df_processed)} samples")
        print(f"Final classes: {df_processed['Group'].value_counts().to_dict()}")
        
        # Shuffle
        df_processed = df_processed.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Distribute among clients
        samples_per_client = len(df_processed) // self.num_clients
        remainder = len(df_processed) % self.num_clients
        
        start_idx = 0
        for i in range(self.num_clients):
            end_idx = start_idx + samples_per_client + (1 if i < remainder else 0)
            client_data = df_processed.iloc[start_idx:end_idx].copy()
            
            client_id = f"Hospital_{chr(65+i)}"
            
            # Create model (7 features: Age, EDUC, SES, eTIV, nWBV, ASF, M/F)
            model = ImprovedAlzheimerNet(input_size=7, hidden_size=64)
            
            # FIXED: Create ImprovedFederatedClient
            client = ImprovedFederatedClient(client_id, client_data, model)
            self.clients.append(client)
            
            start_idx = end_idx
            
            print(f"  {client_id}: {len(client_data)} samples")
            print(f"    Classes: {client_data['Group'].value_counts().to_dict()}")
    
    def get_client_updates(self, epochs: int = 10) -> List[Dict[str, torch.Tensor]]:
        """Get updates from all clients"""
        print(f"\n🔄 Starting enhanced federated training with {len(self.clients)} clients...")
        
        updates = []
        for client in self.clients:
            try:
                client_update = client.local_train(epochs)
                updates.append(client_update)
            except Exception as e:
                print(f"❌ Error training {client.client_id}: {e}")
                # Fallback
                zero_update = {}
                for name, param in client.model.named_parameters():
                    if param.requires_grad:
                        zero_update[name] = torch.zeros_like(param.data)
                updates.append(zero_update)
        
        return updates
    
    def update_all_clients(self, global_gradients: Dict[str, torch.Tensor]):
        """Send updates to all clients"""
        print(f"\n📡 Broadcasting updates to {len(self.clients)} clients...")
        
        for client in self.clients:
            try:
                client.update_model(global_gradients)
            except Exception as e:
                print(f"❌ Error updating {client.client_id}: {e}")
    
    def evaluate_all_clients(self) -> Dict[str, Dict[str, float]]:
        """Evaluate all clients"""
        print(f"\n📈 Enhanced evaluation of all client models...")
        
        results = {}
        total_accuracy = 0
        total_samples = 0
        
        for client in self.clients:
            try:
                metrics = client.evaluate_model()
                results[client.client_id] = metrics
                
                total_accuracy += metrics['accuracy'] * metrics['total']
                total_samples += metrics['total']
                
                print(f"  {client.client_id}: Acc={metrics['accuracy']:.3f}, Loss={metrics['loss']:.4f}")
                
            except Exception as e:
                print(f"❌ Error evaluating {client.client_id}: {e}")
                results[client.client_id] = {
                    'accuracy': 0.0, 'loss': float('inf'), 'correct': 0, 'total': 1
                }
        
        global_accuracy = total_accuracy / total_samples if total_samples > 0 else 0
        results['Global'] = {'accuracy': global_accuracy, 'total_samples': total_samples}
        
        print(f"  🌍 Global Accuracy: {global_accuracy:.3f}")
        
        return results

# Test
if __name__ == "__main__":
    print("🧪 Testing FIXED Federated Learning System")
    
    try:
        manager = ImprovedClientManager('alzheimer.csv', num_clients=3)
        
        updates = manager.get_client_updates(epochs=100)
        print(f"✅ Got updates from {len(updates)} clients")
        
        results = manager.evaluate_all_clients()
        print("✅ Evaluation complete")
        
        print("🎉 FIXED system working!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()