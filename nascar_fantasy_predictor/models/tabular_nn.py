"""PyTorch neural network models for tabular NASCAR data."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class TabularNN(nn.Module):
    """Neural network optimized for tabular NASCAR fantasy data."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = None, 
                 dropout_rate: float = 0.3):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer for fantasy points regression
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x).squeeze(-1)
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class EnsembleTabularNN(nn.Module):
    """Ensemble of tabular neural networks for improved predictions."""
    
    def __init__(self, input_dim: int, num_models: int = 3, 
                 hidden_dims: List[int] = None, dropout_rate: float = 0.3):
        super().__init__()
        
        self.num_models = num_models
        self.models = nn.ModuleList([
            TabularNN(input_dim, hidden_dims, dropout_rate)
            for _ in range(num_models)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble, returning mean prediction."""
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Return mean of all model predictions
        return torch.stack(predictions).mean(dim=0)
    
    def forward_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both mean prediction and uncertainty estimate."""
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred


class AdvancedTabularNN(nn.Module):
    """Advanced neural network with attention mechanism for tabular data."""
    
    def __init__(self, input_dim: int, embedding_dims: Dict[str, int] = None,
                 hidden_dims: List[int] = None, dropout_rate: float = 0.3):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        
        self.input_dim = input_dim
        self.embedding_dims = embedding_dims or {}
        
        # Embedding layers for categorical features
        self.embeddings = nn.ModuleDict()
        embedded_dim = 0
        
        for feature, vocab_size in self.embedding_dims.items():
            embed_dim = min(50, (vocab_size + 1) // 2)
            self.embeddings[feature] = nn.Embedding(vocab_size, embed_dim)
            embedded_dim += embed_dim
        
        # Calculate total input dimension including embeddings
        numerical_dim = input_dim - len(self.embedding_dims)
        total_input_dim = numerical_dim + embedded_dim
        
        # Feature attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(total_input_dim, total_input_dim // 2),
            nn.ReLU(),
            nn.Linear(total_input_dim // 2, total_input_dim),
            nn.Sigmoid()
        )
        
        # Main network
        layers = []
        prev_dim = total_input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layers
        self.main_network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)
        
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor, categorical_features: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional categorical feature embeddings."""
        # Process embeddings if provided
        embedded_features = []
        if categorical_features:
            for feature_name, feature_values in categorical_features.items():
                if feature_name in self.embeddings:
                    embedded = self.embeddings[feature_name](feature_values)
                    embedded_features.append(embedded)
        
        # Combine numerical and embedded features
        if embedded_features:
            embedded_concat = torch.cat(embedded_features, dim=1)
            x = torch.cat([x, embedded_concat], dim=1)
        
        # Apply attention mechanism
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Main network forward pass
        x = self.main_network(x)
        output = self.output_layer(x)
        
        return output.squeeze(-1)
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)


class NASCARPredictor(nn.Module):
    """Specialized NASCAR fantasy points predictor with domain knowledge."""
    
    def __init__(self, input_dim: int, track_type_vocab: int = 6, 
                 manufacturer_vocab: int = 4, hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64, 32]
        
        # Embedding layers for NASCAR-specific categorical features
        self.track_type_embedding = nn.Embedding(track_type_vocab, 8)
        self.manufacturer_embedding = nn.Embedding(manufacturer_vocab, 4)
        
        # Performance branch - focuses on recent performance metrics
        performance_dim = 12  # avg_finish, fantasy_points, etc.
        self.performance_branch = nn.Sequential(
            nn.Linear(performance_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        
        # Speed branch - focuses on speed analytics
        speed_dim = 4  # speed-related features
        self.speed_branch = nn.Sequential(
            nn.Linear(speed_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )
        
        # Track history branch
        track_dim = 8  # track-specific features
        self.track_branch = nn.Sequential(
            nn.Linear(track_dim + 8 + 4, 48),  # +8 for track_type_embedding, +4 for manufacturer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48, 24)
        )
        
        # Combine all branches
        combined_dim = 32 + 16 + 24 + (input_dim - performance_dim - speed_dim - track_dim - 2)  # -2 for categorical
        
        # Final prediction network
        layers = []
        prev_dim = combined_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.final_network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)
        
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through specialized NASCAR predictor."""
        batch_size = x.size(0)
        
        # Extract different feature groups (assuming specific order)
        performance_features = x[:, :12]  # First 12 features
        speed_features = x[:, 12:16]      # Next 4 features
        track_features = x[:, 16:24]      # Next 8 features
        track_type = x[:, 24].long()      # Track type (categorical)
        manufacturer = x[:, 25].long()    # Manufacturer (categorical)
        remaining_features = x[:, 26:]    # Any remaining features
        
        # Process through specialized branches
        perf_out = self.performance_branch(performance_features)
        speed_out = self.speed_branch(speed_features)
        
        # Track branch with embeddings
        track_type_emb = self.track_type_embedding(track_type)
        manufacturer_emb = self.manufacturer_embedding(manufacturer)
        track_input = torch.cat([track_features, track_type_emb, manufacturer_emb], dim=1)
        track_out = self.track_branch(track_input)
        
        # Combine all outputs
        if remaining_features.size(1) > 0:
            combined = torch.cat([perf_out, speed_out, track_out, remaining_features], dim=1)
        else:
            combined = torch.cat([perf_out, speed_out, track_out], dim=1)
        
        # Final prediction
        output = self.final_network(combined)
        return self.output_layer(output).squeeze(-1)
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)