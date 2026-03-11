"""
Physics-Constrained Neural Network (PCNN) for Steel Property Prediction

Key Innovation: Custom loss function enforces UTS > YS constraint
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class PhysicsConstrainedNN(nn.Module):
    """Neural network with physics-aware architecture"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32], dropout_rate: float = 0.3):
        super(PhysicsConstrainedNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers with batch norm and dropout
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Separate heads for each property
        self.yield_strength_head = nn.Linear(prev_dim, 1)
        self.uts_offset_head = nn.Linear(prev_dim, 1)  # Predicts (UTS - YS) to enforce constraint
        self.elongation_head = nn.Linear(prev_dim, 1)
        
    def forward(self, x):
        """Forward pass with constraint enforcement"""
        features = self.feature_extractor(x)
        
        # Predict yield strength directly
        ys = self.yield_strength_head(features)
        
        # Predict UTS as YS + positive offset (ensures UTS > YS)
        uts_offset = torch.abs(self.uts_offset_head(features)) + 1e-6  # Always positive
        uts = ys + uts_offset
        
        # Predict elongation
        elong = torch.abs(self.elongation_head(features))  # Always positive
        
        return torch.cat([ys, uts, elong], dim=1)

class PhysicsConstrainedLoss(nn.Module):
    """Custom loss function with physics constraint penalty"""
    
    def __init__(self, constraint_weight: float = 10.0):
        super(PhysicsConstrainedLoss, self).__init__()
        self.constraint_weight = constraint_weight
        self.mse = nn.MSELoss()
        
    def forward(self, predictions, targets):
        """
        Loss = MSE + constraint_penalty
        
        Args:
            predictions: [batch_size, 3] (YS, UTS, Elongation)
            targets: [batch_size, 3] (YS, UTS, Elongation)
        """
        # Standard MSE loss
        mse_loss = self.mse(predictions, targets)
        
        # Physics constraint: UTS must be > YS
        ys_pred = predictions[:, 0]
        uts_pred = predictions[:, 1]
        
        # Penalty for violations (should be rare due to architecture)
        constraint_violation = torch.relu(ys_pred - uts_pred)  # Only positive when violated
        constraint_loss = torch.mean(constraint_violation ** 2)
        
        total_loss = mse_loss + self.constraint_weight * constraint_loss
        
        return total_loss, mse_loss, constraint_loss

class SteelPropertyPredictor:
    """Main predictor class with training and evaluation"""
    
    def __init__(self, input_dim: int, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = PhysicsConstrainedNN(input_dim).to(self.device)
        self.criterion = PhysicsConstrainedLoss()
        self.scaler_y = None
        self.training_history = {'train_loss': [], 'val_loss': [], 'constraint_violations': []}
        
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
              X_val: pd.DataFrame, y_val: pd.DataFrame,
              epochs: int = 200, batch_size: int = 32, lr: float = 0.001,
              patience: int = 30, verbose: bool = True):
        """Train the model with early stopping"""
        
        # Normalize targets
        from sklearn.preprocessing import StandardScaler
        self.scaler_y = StandardScaler()
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_val_scaled = self.scaler_y.transform(y_val)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train.values).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_scaled).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val.values).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val_scaled).to(self.device)
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=10)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\n{'='*60}")
        print("TRAINING PHYSICS-CONSTRAINED NEURAL NETWORK")
        print(f"{'='*60}")
        print(f"Input features: {X_train.shape[1]}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Device: {self.device}")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            # Mini-batch training
            indices = torch.randperm(len(X_train_tensor))
            for i in range(0, len(X_train_tensor), batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = X_train_tensor[batch_indices]
                y_batch = y_train_tensor[batch_indices]
                
                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss, mse_loss, constraint_loss = self.criterion(predictions, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val_tensor)
                val_loss, val_mse, val_constraint = self.criterion(val_predictions, y_val_tensor)
                
                # Check constraint violations in original scale
                val_pred_original = self.scaler_y.inverse_transform(val_predictions.cpu().numpy())
                violations = np.sum(val_pred_original[:, 0] >= val_pred_original[:, 1])
            
            # Record history
            avg_train_loss = np.mean(train_losses)
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(val_loss.item())
            self.training_history['constraint_violations'].append(violations)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
            
            # Verbose logging
            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Violations: {violations}/{len(X_val)} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            if patience_counter >= patience:
                print(f"\n✓ Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        print(f"✓ Training completed. Best validation loss: {best_val_loss:.4f}")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values).to(self.device)
            predictions_scaled = self.model(X_tensor).cpu().numpy()
            predictions = self.scaler_y.inverse_transform(predictions_scaled)
        
        return predictions
    
    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> dict:
        """Comprehensive evaluation"""
        predictions = self.predict(X)
        y_true = y.values
        
        metrics = {}
        target_names = ['Yield Strength', 'UTS', 'Elongation']
        
        print(f"\n{'='*60}")
        print("MODEL EVALUATION METRICS")
        print(f"{'='*60}")
        
        for i, name in enumerate(target_names):
            rmse = np.sqrt(mean_squared_error(y_true[:, i], predictions[:, i]))
            mae = mean_absolute_error(y_true[:, i], predictions[:, i])
            r2 = r2_score(y_true[:, i], predictions[:, i])
            
            metrics[f'{name}_RMSE'] = rmse
            metrics[f'{name}_MAE'] = mae
            metrics[f'{name}_R2'] = r2
            
            print(f"\n{name}:")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE:  {mae:.2f}")
            print(f"  R²:   {r2:.4f}")
        
        # Check constraint violations
        violations = np.sum(predictions[:, 0] >= predictions[:, 1])
        metrics['Constraint_Violations'] = violations
        metrics['Violation_Rate'] = violations / len(predictions)
        
        print(f"\n{'='*60}")
        print(f"PHYSICS CONSTRAINT VALIDATION")
        print(f"{'='*60}")
        print(f"Violations (UTS ≤ YS): {violations}/{len(predictions)} ({100*violations/len(predictions):.2f}%)")
        
        if violations == 0:
            print("✓ PERFECT! No constraint violations!")
        
        return metrics
    
    def save_model(self, path: str = "models/pcnn_model.pt"):
        """Save model and scaler"""
        import joblib
        torch.save(self.model.state_dict(), path)
        joblib.dump(self.scaler_y, path.replace('.pt', '_scaler.pkl'))
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path: str = "models/pcnn_model.pt"):
        """Load model and scaler"""
        import joblib
        self.model.load_state_dict(torch.load(path))
        self.scaler_y = joblib.load(path.replace('.pt', '_scaler.pkl'))
        print(f"✓ Model loaded from {path}")
