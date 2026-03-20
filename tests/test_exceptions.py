"""
Unit tests for exceptions module.
"""
import pytest

from src.exceptions import (
    SteelMLException,
    InvalidCompositionError,
    ModelNotTrainedError,
    ModelLoadError,
    DataValidationError,
    FeatureEngineeringError,
    OptimizationError,
    PhysicsConstraintViolation,
    ConfigurationError
)


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""
    
    def test_base_exception(self):
        """Test base SteelMLException."""
        exc = SteelMLException("Test error")
        assert isinstance(exc, Exception)
        assert str(exc) == "Test error"
    
    def test_invalid_composition_error(self):
        """Test InvalidCompositionError."""
        composition = {'C': 0.5, 'Mn': 0.8}
        exc = InvalidCompositionError("Invalid composition", composition=composition)
        assert isinstance(exc, SteelMLException)
        assert exc.composition == composition
    
    def test_model_not_trained_error(self):
        """Test ModelNotTrainedError."""
        exc = ModelNotTrainedError("Model not trained")
        assert isinstance(exc, SteelMLException)
    
    def test_model_load_error(self):
        """Test ModelLoadError."""
        path = "/path/to/model.pt"
        exc = ModelLoadError("Failed to load", path=path)
        assert isinstance(exc, SteelMLException)
        assert exc.path == path
    
    def test_data_validation_error(self):
        """Test DataValidationError."""
        exc = DataValidationError("Data validation failed")
        assert isinstance(exc, SteelMLException)
    
    def test_feature_engineering_error(self):
        """Test FeatureEngineeringError."""
        exc = FeatureEngineeringError("Feature engineering failed")
        assert isinstance(exc, SteelMLException)
    
    def test_optimization_error(self):
        """Test OptimizationError."""
        exc = OptimizationError("Optimization failed")
        assert isinstance(exc, SteelMLException)
    
    def test_physics_constraint_violation(self):
        """Test PhysicsConstraintViolation."""
        exc = PhysicsConstraintViolation("UTS < YS", constraint_type="UTS_GT_YS")
        assert isinstance(exc, SteelMLException)
        assert exc.constraint_type == "UTS_GT_YS"
    
    def test_configuration_error(self):
        """Test ConfigurationError."""
        exc = ConfigurationError("Invalid configuration")
        assert isinstance(exc, SteelMLException)


class TestExceptionUsage:
    """Tests for exception usage patterns."""
    
    def test_raise_and_catch_invalid_composition(self):
        """Test raising and catching InvalidCompositionError."""
        with pytest.raises(InvalidCompositionError) as exc_info:
            raise InvalidCompositionError("Total exceeds 100%")
        
        assert "Total exceeds 100%" in str(exc_info.value)
    
    def test_raise_and_catch_model_not_trained(self):
        """Test raising and catching ModelNotTrainedError."""
        with pytest.raises(ModelNotTrainedError):
            raise ModelNotTrainedError("Cannot predict without training")
    
    def test_catch_base_exception(self):
        """Test that all custom exceptions can be caught as SteelMLException."""
        with pytest.raises(SteelMLException):
            raise InvalidCompositionError("Test")
        
        with pytest.raises(SteelMLException):
            raise ModelLoadError("Test")
        
        with pytest.raises(SteelMLException):
            raise PhysicsConstraintViolation("Test")
