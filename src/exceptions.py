"""
Custom exceptions for SteelML.
Provides specific exception types for better error handling and debugging.
"""


class SteelMLException(Exception):
    """Base exception for all SteelML errors."""
    pass


class InvalidCompositionError(SteelMLException):
    """Raised when a steel composition is physically invalid."""
    
    def __init__(self, message: str, composition: dict = None):
        self.composition = composition
        super().__init__(message)


class ModelNotTrainedError(SteelMLException):
    """Raised when attempting to use a model that hasn't been trained."""
    pass


class ModelLoadError(SteelMLException):
    """Raised when model loading fails."""
    
    def __init__(self, message: str, path: str = None):
        self.path = path
        super().__init__(message)


class DataValidationError(SteelMLException):
    """Raised when data validation fails."""
    pass


class FeatureEngineeringError(SteelMLException):
    """Raised when feature engineering fails."""
    pass


class OptimizationError(SteelMLException):
    """Raised when inverse design optimization fails."""
    pass


class PhysicsConstraintViolation(SteelMLException):
    """Raised when physics constraints are violated."""
    
    def __init__(self, message: str, constraint_type: str = None):
        self.constraint_type = constraint_type
        super().__init__(message)


class ConfigurationError(SteelMLException):
    """Raised when configuration is invalid."""
    pass


__all__ = [
    'SteelMLException',
    'InvalidCompositionError',
    'ModelNotTrainedError',
    'ModelLoadError',
    'DataValidationError',
    'FeatureEngineeringError',
    'OptimizationError',
    'PhysicsConstraintViolation',
    'ConfigurationError',
]
