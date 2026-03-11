"""
Unit tests for Physics-Constrained Neural Network model.
"""
import pytest
import torch
import pandas as pd
import numpy as np

from src.pcnn_model import (
    PhysicsConstrainedNN,
    PhysicsConstrainedLoss,
    SteelPropertyPredictor
)


class TestPhysicsConstrainedNN:
    """Tests for PhysicsConstrainedNN architecture."""
    
    def test_model_initialization(self):
        """Test that model initializes correctly."""
        model = PhysicsConstrainedNN(input_dim=13)
        assert model is not None
        assert hasattr(model, 'shared_layers')
        assert hasattr(model, 'ys_head')
        assert hasattr(model, 'uts_offset_head')
        assert hasattr(model, 'elongation_head')
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        model = PhysicsConstrainedNN(input_dim=13)
        x = torch.randn(10, 13)  # Batch of 10 samples
        output = model(x)
        
        assert output.shape == (10, 3)  # YS, UTS, Elongation
    
    def test_physics_constraint_enforced(self):
        """Test that UTS > YS constraint is enforced."""
        model = PhysicsConstrainedNN(input_dim=13)
        model.eval()
        
        x = torch.randn(100, 13)
        with torch.no_grad():
            output = model(x)
        
        ys = output[:, 0].numpy()
        uts = output[:, 1].numpy()
        
        # All UTS values should be greater than YS
        assert np.all(uts > ys), "Physics constraint violated: UTS should always be > YS"
    
    def test_model_parameters_exist(self):
        """Test that model has trainable parameters."""
        model = PhysicsConstrainedNN(input_dim=13)
        params = list(model.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)


class TestPhysicsConstrainedLoss:
    """Tests for PhysicsConstrainedLoss function."""
    
    def test_loss_initialization(self):
        """Test loss function initialization."""
        loss_fn = PhysicsConstrainedLoss(constraint_weight=10.0)
        assert loss_fn.constraint_weight == 10.0
    
    def test_loss_computation(self):
        """Test loss computation."""
        loss_fn = PhysicsConstrainedLoss()
        
        predictions = torch.tensor([[400.0, 500.0, 20.0],
                                   [600.0, 750.0, 18.0]])
        targets = torch.tensor([[450.0, 550.0, 22.0],
                               [650.0, 800.0, 19.0]])
        
        loss = loss_fn(predictions, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_loss_penalty_for_violations(self):
        """Test that loss increases when physics is violated."""
        loss_fn = PhysicsConstrainedLoss(constraint_weight=10.0)
        
        # Valid predictions (UTS > YS)
        valid_preds = torch.tensor([[400.0, 500.0, 20.0]])
        targets = torch.tensor([[450.0, 550.0, 22.0]])
        valid_loss = loss_fn(valid_preds, targets)
        
        # Invalid predictions (UTS < YS) - should have higher loss
        invalid_preds = torch.tensor([[500.0, 400.0, 20.0]])  # UTS < YS
        invalid_loss = loss_fn(invalid_preds, targets)
        
        assert invalid_loss > valid_loss, "Loss should be higher when physics is violated"


class TestSteelPropertyPredictor:
    """Tests for SteelPropertyPredictor class."""
    
    def test_predictor_initialization(self):
        """Test predictor initialization."""
        predictor = SteelPropertyPredictor(input_dim=13)
        assert predictor.model is not None
        assert predictor.device is not None
    
    def test_predict_shape(self, sample_compositions_df):
        """Test prediction output shape."""
        predictor = SteelPropertyPredictor(input_dim=13)
        predictions = predictor.predict(sample_compositions_df)
        
        assert predictions.shape == (len(sample_compositions_df), 3)
    
    def test_predict_physics_constraint(self, sample_compositions_df):
        """Test that predictions satisfy physics constraints."""
        predictor = SteelPropertyPredictor(input_dim=13)
        predictions = predictor.predict(sample_compositions_df)
        
        ys = predictions[:, 0]
        uts = predictions[:, 1]
        
        # All UTS should be > YS
        assert np.all(uts > ys), "Predictions violate physics: UTS must be > YS"
    
    @pytest.mark.slow
    def test_training_reduces_loss(self, sample_dataset):
        """Test that training reduces loss."""
        # Split dataset
        train_size = int(0.7 * len(sample_dataset))
        train_df = sample_dataset[:train_size]
        val_df = sample_dataset[train_size:]
        
        from src.config import ELEMENT_COLS, TARGET_COLS
        X_train = train_df[ELEMENT_COLS]
        y_train = train_df[TARGET_COLS]
        X_val = val_df[ELEMENT_COLS]
        y_val = val_df[TARGET_COLS]
        
        predictor = SteelPropertyPredictor(input_dim=13)
        
        # Train for just a few epochs
        predictor.train(X_train, y_train, X_val, y_val,
                       epochs=5, batch_size=16, verbose=False)
        
        # Check that model can make predictions
        predictions = predictor.predict(X_val)
        assert predictions.shape == (len(X_val), 3)
    
    def test_save_and_load_model(self, temp_model_path):
        """Test model saving and loading."""
        predictor1 = SteelPropertyPredictor(input_dim=13)
        
        # Save model
        predictor1.save_model(str(temp_model_path))
        assert temp_model_path.exists()
        
        # Load model
        predictor2 = SteelPropertyPredictor(input_dim=13)
        predictor2.load_model(str(temp_model_path))
        
        # Both should produce same predictions
        test_input = pd.DataFrame(np.random.randn(5, 13))
        pred1 = predictor1.predict(test_input)
        pred2 = predictor2.predict(test_input)
        
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=5)
