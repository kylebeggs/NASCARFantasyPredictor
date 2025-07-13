"""Custom exceptions for NASCAR Fantasy Predictor."""


class NASCARPredictorError(Exception):
    """Base exception for NASCAR Fantasy Predictor."""
    pass


class DataError(NASCARPredictorError):
    """Exception raised for data-related errors."""
    pass


class ModelError(NASCARPredictorError):
    """Exception raised for model-related errors."""
    pass


class PredictionError(NASCARPredictorError):
    """Exception raised for prediction-related errors."""
    pass


class FeatureEngineeringError(NASCARPredictorError):
    """Exception raised for feature engineering errors."""
    pass


class ScrapingError(DataError):
    """Exception raised for web scraping errors."""
    pass


class ValidationError(NASCARPredictorError):
    """Exception raised for validation errors."""
    pass


class ConfigurationError(NASCARPredictorError):
    """Exception raised for configuration errors."""
    pass