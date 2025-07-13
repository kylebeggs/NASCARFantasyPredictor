"""Feature importance analysis for NASCAR fantasy predictions."""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class PyTorchWrapper(BaseEstimator):
    """Wrapper to make PyTorch model compatible with sklearn's permutation_importance."""
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.eval()
    
    def predict(self, X):
        """Predict method for sklearn compatibility."""
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X)
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.cpu().numpy()
        
        return predictions


class FeatureImportanceAnalyzer:
    """Comprehensive feature importance analysis for NASCAR prediction models."""
    
    def __init__(self, model: nn.Module, feature_names: List[str], device: torch.device = None):
        """
        Initialize the analyzer.
        
        Args:
            model: Trained PyTorch model
            feature_names: List of feature names in order
            device: Device to run calculations on
        """
        self.model = model
        self.feature_names = feature_names
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize SHAP explainer if available
        self.shap_explainer = None
        if SHAP_AVAILABLE:
            self._initialize_shap()
    
    def _initialize_shap(self):
        """Initialize SHAP explainer."""
        try:
            # Create a wrapper function for SHAP
            def model_predict(x):
                if isinstance(x, np.ndarray):
                    x = torch.FloatTensor(x).to(self.device)
                with torch.no_grad():
                    return self.model(x).cpu().numpy()
            
            self.shap_explainer = shap.Explainer(model_predict)
        except Exception as e:
            warnings.warn(f"Could not initialize SHAP explainer: {e}")
    
    def get_weight_importance(self) -> Dict[str, float]:
        """
        Analyze feature importance based on first layer weights.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Get first linear layer
        first_layer = None
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                first_layer = module
                break
        
        if first_layer is None:
            raise ValueError("No linear layers found in model")
        
        # Calculate importance as mean absolute weight per input feature
        weights = first_layer.weight.data.cpu().numpy()  # Shape: [hidden_dim, input_dim]
        feature_importance = np.mean(np.abs(weights), axis=0)  # Average across hidden units
        
        # Normalize to sum to 1
        feature_importance = feature_importance / np.sum(feature_importance)
        
        return dict(zip(self.feature_names, feature_importance))
    
    def get_gradient_importance(self, X: torch.Tensor, y: torch.Tensor = None) -> Dict[str, float]:
        """
        Calculate feature importance using input gradients.
        
        Args:
            X: Input tensor [batch_size, num_features]
            y: Optional target tensor for guided gradients
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        X = X.to(self.device)
        X.requires_grad_(True)
        
        # Forward pass
        predictions = self.model(X)
        
        # Calculate gradients
        if y is not None:
            y = y.to(self.device)
            loss = nn.MSELoss()(predictions, y)
            gradients = torch.autograd.grad(loss, X, create_graph=False)[0]
        else:
            # Use output as target (gradient of output w.r.t. input)
            gradients = torch.autograd.grad(
                predictions.sum(), X, create_graph=False
            )[0]
        
        # Calculate importance as mean absolute gradient
        feature_importance = torch.mean(torch.abs(gradients), dim=0).cpu().numpy()
        
        # Normalize
        feature_importance = feature_importance / np.sum(feature_importance)
        
        return dict(zip(self.feature_names, feature_importance))
    
    def get_permutation_importance(self, X: np.ndarray, y: np.ndarray, 
                                 n_repeats: int = 10) -> Dict[str, float]:
        """
        Calculate permutation feature importance.
        
        Args:
            X: Input data [n_samples, n_features]
            y: Target values [n_samples]
            n_repeats: Number of permutation repeats
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Create sklearn-compatible wrapper
        wrapped_model = PyTorchWrapper(self.model, self.device)
        
        # Calculate permutation importance
        result = permutation_importance(
            wrapped_model, X, y, 
            n_repeats=n_repeats, 
            random_state=42,
            scoring='neg_mean_squared_error'
        )
        
        # Extract importance scores
        importance_scores = result.importances_mean
        
        # Normalize
        importance_scores = importance_scores / np.sum(np.abs(importance_scores))
        
        return dict(zip(self.feature_names, importance_scores))
    
    def get_shap_importance(self, X: np.ndarray, background_samples: int = 100) -> Dict[str, float]:
        """
        Calculate SHAP feature importance.
        
        Args:
            X: Input data [n_samples, n_features]
            background_samples: Number of background samples for SHAP
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            raise ValueError("SHAP not available or not initialized")
        
        # Use subset as background if X is large
        if len(X) > background_samples:
            background_indices = np.random.choice(len(X), background_samples, replace=False)
            background_data = X[background_indices]
        else:
            background_data = X
        
        # Calculate SHAP values
        shap_values = self.shap_explainer(background_data)
        
        # Calculate importance as mean absolute SHAP value
        if hasattr(shap_values, 'values'):
            importance_scores = np.mean(np.abs(shap_values.values), axis=0)
        else:
            importance_scores = np.mean(np.abs(shap_values), axis=0)
        
        # Normalize
        importance_scores = importance_scores / np.sum(importance_scores)
        
        return dict(zip(self.feature_names, importance_scores))
    
    def get_comprehensive_importance(self, X: np.ndarray, y: np.ndarray, 
                                   methods: List[str] = None) -> pd.DataFrame:
        """
        Calculate feature importance using multiple methods.
        
        Args:
            X: Input data [n_samples, n_features]
            y: Target values [n_samples]
            methods: List of methods to use ['weights', 'gradients', 'permutation', 'shap']
            
        Returns:
            DataFrame with features as rows and methods as columns
        """
        if methods is None:
            methods = ['weights', 'gradients', 'permutation']
            if SHAP_AVAILABLE and self.shap_explainer is not None:
                methods.append('shap')
        
        results = {}
        
        # Weight-based importance
        if 'weights' in methods:
            results['weights'] = self.get_weight_importance()
        
        # Gradient-based importance
        if 'gradients' in methods:
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)
            results['gradients'] = self.get_gradient_importance(X_tensor, y_tensor)
        
        # Permutation importance
        if 'permutation' in methods:
            results['permutation'] = self.get_permutation_importance(X, y)
        
        # SHAP importance
        if 'shap' in methods and SHAP_AVAILABLE and self.shap_explainer is not None:
            try:
                results['shap'] = self.get_shap_importance(X)
            except Exception as e:
                warnings.warn(f"SHAP calculation failed: {e}")
        
        # Create DataFrame
        importance_df = pd.DataFrame(results, index=self.feature_names)
        
        # Calculate average importance across methods
        importance_df['average'] = importance_df.mean(axis=1)
        
        # Sort by average importance
        importance_df = importance_df.sort_values('average', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, 
                              top_k: int = 15, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot feature importance comparison across methods.
        
        Args:
            importance_df: DataFrame from get_comprehensive_importance
            top_k: Number of top features to show
            figsize: Figure size
        """
        # Get top k features by average importance
        top_features = importance_df.head(top_k)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Feature Importance Analysis', fontsize=16)
        
        # Plot 1: Bar plot of average importance
        ax1 = axes[0, 0]
        top_features['average'].plot(kind='barh', ax=ax1)
        ax1.set_title('Average Feature Importance')
        ax1.set_xlabel('Importance Score')
        
        # Plot 2: Heatmap of all methods
        ax2 = axes[0, 1]
        method_cols = [col for col in top_features.columns if col != 'average']
        if method_cols:
            sns.heatmap(top_features[method_cols].T, annot=True, fmt='.3f', 
                       cmap='YlOrRd', ax=ax2)
            ax2.set_title('Importance by Method')
        
        # Plot 3: Method comparison for top 5 features
        ax3 = axes[1, 0]
        if len(method_cols) > 1:
            top_5 = top_features.head(5)[method_cols]
            top_5.plot(kind='bar', ax=ax3)
            ax3.set_title('Top 5 Features - Method Comparison')
            ax3.set_ylabel('Importance Score')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 4: Feature importance distribution
        ax4 = axes[1, 1]
        top_features['average'].hist(bins=10, ax=ax4)
        ax4.set_title('Importance Score Distribution')
        ax4.set_xlabel('Importance Score')
        ax4.set_ylabel('Count')
        
        plt.tight_layout()
        return fig
    
    def explain_prediction(self, X: np.ndarray, driver_names: List[str] = None,
                          method: str = 'gradients') -> pd.DataFrame:
        """
        Explain individual predictions.
        
        Args:
            X: Input data for specific drivers [n_drivers, n_features]
            driver_names: Names of drivers (optional)
            method: Method to use for explanation
            
        Returns:
            DataFrame with driver explanations
        """
        if driver_names is None:
            driver_names = [f"Driver_{i}" for i in range(len(X))]
        
        explanations = []
        
        for i, (driver_data, driver_name) in enumerate(zip(X, driver_names)):
            driver_data = driver_data.reshape(1, -1)  # Add batch dimension
            
            if method == 'gradients':
                X_tensor = torch.FloatTensor(driver_data)
                importance = self.get_gradient_importance(X_tensor)
            elif method == 'shap' and SHAP_AVAILABLE and self.shap_explainer is not None:
                importance = self.get_shap_importance(driver_data)
            else:
                raise ValueError(f"Method {method} not supported for individual explanations")
            
            # Get prediction
            with torch.no_grad():
                prediction = self.model(torch.FloatTensor(driver_data).to(self.device))
                prediction = prediction.cpu().numpy()[0]
            
            explanation = {
                'driver': driver_name,
                'prediction': prediction,
                **importance
            }
            explanations.append(explanation)
        
        return pd.DataFrame(explanations)
    
    def get_top_features(self, importance_df: pd.DataFrame, n: int = 10) -> List[str]:
        """Get names of top N most important features."""
        return importance_df.head(n).index.tolist()
    
    def get_track_specific_importance(self, X: np.ndarray, y: np.ndarray, 
                                    track_data: np.ndarray, track_name: str,
                                    methods: List[str] = None) -> pd.DataFrame:
        """
        Calculate feature importance for a specific track.
        
        Args:
            X: Full input data [n_samples, n_features]
            y: Full target values [n_samples]
            track_data: Track indicators (1 for target track, 0 for others) [n_samples]
            track_name: Name of the track for reporting
            methods: List of methods to use
            
        Returns:
            DataFrame with track-specific feature importance
        """
        # Filter to only data from the specified track
        track_mask = track_data == 1
        if not np.any(track_mask):
            raise ValueError(f"No data found for track: {track_name}")
        
        X_track = X[track_mask]
        y_track = y[track_mask]
        
        print(f"Analyzing {len(X_track)} samples from {track_name}")
        
        # Get importance for this track's data
        importance_df = self.get_comprehensive_importance(X_track, y_track, methods)
        
        # Add track info to the results
        importance_df.attrs = {'track_name': track_name, 'sample_count': len(X_track)}
        
        return importance_df
    
    def compare_track_importance(self, X: np.ndarray, y: np.ndarray,
                               track_data: np.ndarray, track_names: List[str],
                               track_indicators: List[int],
                               methods: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Compare feature importance across multiple tracks.
        
        Args:
            X: Full input data [n_samples, n_features]
            y: Full target values [n_samples]
            track_data: Track type indicators [n_samples]
            track_names: Names of tracks to compare
            track_indicators: Corresponding indicator values for each track
            methods: Methods to use for analysis
            
        Returns:
            Dictionary mapping track names to their importance DataFrames
        """
        track_importance = {}
        
        for track_name, track_id in zip(track_names, track_indicators):
            try:
                track_mask = track_data == track_id
                if np.sum(track_mask) >= 20:  # Minimum samples needed
                    importance_df = self.get_track_specific_importance(
                        X, y, track_data == track_id, track_name, methods
                    )
                    track_importance[track_name] = importance_df
                else:
                    print(f"Skipping {track_name}: insufficient data ({np.sum(track_mask)} samples)")
            except Exception as e:
                print(f"Error analyzing {track_name}: {e}")
        
        return track_importance
    
    def plot_track_comparison(self, track_importance: Dict[str, pd.DataFrame],
                            top_k: int = 10, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot feature importance comparison across tracks.
        
        Args:
            track_importance: Dictionary from compare_track_importance
            top_k: Number of top features to show
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Feature Importance by Track Type (Top {top_k} Features)', fontsize=16)
        
        # Collect all features and their importance across tracks
        all_features = set()
        for df in track_importance.values():
            all_features.update(df.head(top_k).index)
        
        # Create comparison matrix
        comparison_data = []
        for feature in all_features:
            row = {'feature': feature}
            for track_name, df in track_importance.items():
                if feature in df.index:
                    row[track_name] = df.loc[feature, 'average']
                else:
                    row[track_name] = 0.0
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data).set_index('feature')
        
        # Plot 1: Heatmap of feature importance by track
        ax1 = axes[0, 0]
        sns.heatmap(comparison_df.T, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax1)
        ax1.set_title('Feature Importance Heatmap')
        ax1.set_xlabel('Features')
        ax1.set_ylabel('Tracks')
        
        # Plot 2: Top features for first track
        if track_importance:
            first_track = list(track_importance.keys())[0]
            ax2 = axes[0, 1]
            track_importance[first_track].head(top_k)['average'].plot(kind='barh', ax=ax2)
            ax2.set_title(f'Top Features - {first_track}')
            ax2.set_xlabel('Importance')
        
        # Plot 3: Feature importance correlation between tracks
        if len(track_importance) >= 2:
            ax3 = axes[1, 0]
            corr_matrix = comparison_df.corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax3)
            ax3.set_title('Track Correlation')
        
        # Plot 4: Feature variance across tracks
        ax4 = axes[1, 1]
        feature_variance = comparison_df.var(axis=1).sort_values(ascending=False)
        feature_variance.head(top_k).plot(kind='bar', ax=ax4)
        ax4.set_title('Feature Importance Variance Across Tracks')
        ax4.set_xlabel('Features')
        ax4.set_ylabel('Variance')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig

    def save_importance_report(self, importance_df: pd.DataFrame, 
                             output_path: str, include_plot: bool = True):
        """
        Save comprehensive importance report.
        
        Args:
            importance_df: DataFrame from get_comprehensive_importance
            output_path: Path to save report (without extension)
            include_plot: Whether to include visualization
        """
        # Check if this is track-specific analysis
        track_name = getattr(importance_df, 'attrs', {}).get('track_name', 'All Tracks')
        sample_count = getattr(importance_df, 'attrs', {}).get('sample_count', len(importance_df))
        
        # Save CSV
        importance_df.to_csv(f"{output_path}.csv")
        
        # Save detailed report
        with open(f"{output_path}.txt", 'w') as f:
            f.write(f"NASCAR Fantasy Predictor - Feature Importance Report\n")
            f.write(f"Track: {track_name}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Top 10 Most Important Features:\n")
            f.write("-" * 30 + "\n")
            for i, (feature, row) in enumerate(importance_df.head(10).iterrows(), 1):
                f.write(f"{i:2d}. {feature:<25} (avg: {row['average']:.4f})\n")
            
            f.write(f"\nFeature Importance Summary:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Track analyzed: {track_name}\n")
            f.write(f"Sample count: {sample_count}\n")
            f.write(f"Total features analyzed: {len(importance_df)}\n")
            f.write(f"Top 10 features account for: {importance_df.head(10)['average'].sum():.1%} of total importance\n")
            
            # Method comparison
            method_cols = [col for col in importance_df.columns if col != 'average']
            if len(method_cols) > 1:
                f.write(f"\nMethod Correlation:\n")
                f.write("-" * 20 + "\n")
                corr_matrix = importance_df[method_cols].corr()
                f.write(corr_matrix.to_string())
        
        # Save plot
        if include_plot:
            fig = self.plot_feature_importance(importance_df)
            fig.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        print(f"Importance report saved to {output_path}.*")