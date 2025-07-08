"""PyTorch model training and management."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
from pathlib import Path
import json
from datetime import datetime

from .tabular_nn import TabularNN


class ModelTrainer:
    """Trainer for NASCAR fantasy point prediction models."""
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.training_history = []
        
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'finish_position',
                    feature_columns: List[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for training."""
        if feature_columns is None:
            # Exclude non-feature columns
            exclude_cols = ['driver_id', 'driver_name', 'race_id', 'date', target_column, 'fantasy_points']
            feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        self.feature_columns = feature_columns
        
        # Extract features and target
        X = df[feature_columns].fillna(0).values
        y = df[target_column].fillna(0).values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        return X_tensor, y_tensor
    
    def create_model(self, input_dim: int, **kwargs) -> nn.Module:
        """Create tabular neural network model."""
        model = TabularNN(input_dim, **kwargs)
        return model.to(self.device)
    
    def train(self, X: torch.Tensor, y: torch.Tensor, 
              validation_split: float = 0.2, epochs: int = 100,
              batch_size: int = 64, learning_rate: float = 0.001,
              early_stopping_patience: int = 10, **model_kwargs) -> Dict[str, Any]:
        """Train the model with validation."""
        
        # Create model
        input_dim = X.shape[1]
        self.model = self.create_model(input_dim, **model_kwargs)
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Load best model state
        self.model.load_state_dict(best_model_state)
        
        # Calculate final metrics
        train_mae = self.evaluate(X_train, y_train, metric='mae')
        val_mae = self.evaluate(X_val, y_val, metric='mae')
        
        training_result = {
            'final_train_loss': train_losses[-1],
            'final_val_loss': best_val_loss,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'epochs_trained': len(train_losses),
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        self.training_history.append(training_result)
        return training_result
    
    def cross_validate(self, X: torch.Tensor, y: torch.Tensor, 
                      n_splits: int = 5, **train_kwargs) -> Dict[str, Any]:
        """Perform time series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = []
        
        # Convert back to numpy for sklearn compatibility
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_np)):
            print(f"Training fold {fold + 1}/{n_splits}")
            
            X_train_fold = torch.FloatTensor(X_np[train_idx]).to(self.device)
            y_train_fold = torch.FloatTensor(y_np[train_idx]).to(self.device)
            X_val_fold = torch.FloatTensor(X_np[val_idx]).to(self.device)
            y_val_fold = torch.FloatTensor(y_np[val_idx]).to(self.device)
            
            # Train model for this fold
            fold_result = self.train(
                torch.cat([X_train_fold, X_val_fold]), 
                torch.cat([y_train_fold, y_val_fold]),
                validation_split=len(X_val_fold) / (len(X_train_fold) + len(X_val_fold)),
                **train_kwargs
            )
            
            cv_results.append(fold_result)
        
        # Aggregate results
        avg_val_mae = np.mean([r['val_mae'] for r in cv_results])
        std_val_mae = np.std([r['val_mae'] for r in cv_results])
        
        return {
            'cv_results': cv_results,
            'avg_val_mae': avg_val_mae,
            'std_val_mae': std_val_mae
        }
    
    def incremental_train(self, X_new: torch.Tensor, y_new: torch.Tensor,
                         epochs: int = 10, learning_rate: float = 0.0001):
        """Perform incremental training with new data."""
        if self.model is None:
            raise ValueError("No model to update. Train a model first.")
        
        # Create data loader for new data
        dataset = TensorDataset(X_new, y_new)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Use lower learning rate for incremental updates
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 5 == 0:
                print(f"Incremental training epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """Make predictions with the trained model."""
        if self.model is None:
            raise ValueError("No trained model available.")
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
            return predictions.cpu().numpy()
    
    def predict_with_uncertainty(self, X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates using Monte Carlo dropout."""
        # Use dropout for uncertainty estimation
        self.model.train()  # Enable dropout
        predictions = []
        
        with torch.no_grad():
            for _ in range(10):  # Monte Carlo sampling
                pred = self.model(X)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        return mean_pred, std_pred
    
    def evaluate(self, X: torch.Tensor, y: torch.Tensor, metric: str = 'mse') -> float:
        """Evaluate model performance."""
        predictions = self.predict(X)
        y_true = y.cpu().numpy()
        
        if metric == 'mse':
            return np.mean((predictions - y_true) ** 2)
        elif metric == 'mae':
            return np.mean(np.abs(predictions - y_true))
        elif metric == 'rmse':
            return np.sqrt(np.mean((predictions - y_true) ** 2))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def save_model(self, filepath: str):
        """Save trained model and scaler."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        # Get actual input dimension from model's first layer
        if hasattr(self.model, 'input_dim'):
            actual_input_dim = self.model.input_dim
        elif hasattr(self.model, 'final_network') and len(self.model.final_network) > 0:
            actual_input_dim = self.model.final_network[0].in_features
        else:
            actual_input_dim = len(self.feature_columns) if self.feature_columns else None
            
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': actual_input_dim,
            'feature_columns': self.feature_columns,
            'device': self.device,
            'training_history': self.training_history
        }, filepath.with_suffix('.pth'))
        
        # Save scaler
        joblib.dump(self.scaler, filepath.with_suffix('.scaler'))
        
        # Save metadata
        metadata = {
            'saved_at': datetime.now().isoformat(),
            'model_type': 'tabular',
            'feature_count': len(self.feature_columns) if self.feature_columns else None,
            'training_history': self._convert_to_serializable(self.training_history)
        }
        
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to JSON-serializable types."""
        import numpy as np
        
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        else:
            return obj
    
    def load_model(self, filepath: str):
        """Load trained model and scaler."""
        filepath = Path(filepath)
        
        # Load model
        checkpoint = torch.load(filepath.with_suffix('.pth'), map_location=self.device, weights_only=False)
        
        self.feature_columns = checkpoint['feature_columns']
        self.training_history = checkpoint.get('training_history', [])
        
        # Recreate model
        input_dim = checkpoint.get('input_dim')
        if input_dim is None:
            input_dim = len(self.feature_columns) if self.feature_columns else 30
        self.model = self.create_model(input_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load scaler
        self.scaler = joblib.load(filepath.with_suffix('.scaler'))
        
        print(f"Model loaded successfully from {filepath}")