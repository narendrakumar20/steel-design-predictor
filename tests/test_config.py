"""
Unit tests for configuration module.
"""
import pytest
from pathlib import Path

from src.config import (
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    LOGS_DIR,
    ELEMENT_COLS,
    TARGET_COLS,
    ModelConfig,
    DataConfig,
    InverseDesignConfig,
    LoggingConfig,
    AppConfig
)


class TestPaths:
    """Tests for path configurations."""
    
    def test_project_root_exists(self):
        """Test that project root is a valid path."""
        assert isinstance(PROJECT_ROOT, Path)
        assert PROJECT_ROOT.exists()
    
    def test_directories_are_paths(self):
        """Test that all directory configs are Path objects."""
        assert isinstance(DATA_DIR, Path)
        assert isinstance(MODELS_DIR, Path)
        assert isinstance(PLOTS_DIR, Path)
        assert isinstance(LOGS_DIR, Path)
    
    def test_directories_created(self):
        """Test that directories are created."""
        # These should be created by config module
        assert DATA_DIR.exists()
        assert MODELS_DIR.exists()
        assert PLOTS_DIR.exists()
        assert LOGS_DIR.exists()


class TestConstants:
    """Tests for constant definitions."""
    
    def test_element_cols(self):
        """Test element columns definition."""
        assert isinstance(ELEMENT_COLS, list)
        assert len(ELEMENT_COLS) == 13
        assert all(isinstance(elem, str) for elem in ELEMENT_COLS)
        assert 'C' in ELEMENT_COLS
        assert 'Mn' in ELEMENT_COLS
    
    def test_target_cols(self):
        """Test target columns definition."""
        assert isinstance(TARGET_COLS, list)
        assert len(TARGET_COLS) == 3
        assert 'yield_strength_MPa' in TARGET_COLS
        assert 'ultimate_tensile_strength_MPa' in TARGET_COLS
        assert 'elongation_percent' in TARGET_COLS


class TestModelConfig:
    """Tests for ModelConfig class."""
    
    def test_hidden_dims(self):
        """Test hidden dimensions configuration."""
        assert isinstance(ModelConfig.HIDDEN_DIMS, list)
        assert len(ModelConfig.HIDDEN_DIMS) > 0
        assert all(isinstance(dim, int) for dim in ModelConfig.HIDDEN_DIMS)
    
    def test_dropout_rate(self):
        """Test dropout rate configuration."""
        assert isinstance(ModelConfig.DROPOUT_RATE, float)
        assert 0 <= ModelConfig.DROPOUT_RATE <= 1
    
    def test_training_params(self):
        """Test training parameter configurations."""
        assert isinstance(ModelConfig.EPOCHS, int)
        assert ModelConfig.EPOCHS > 0
        assert isinstance(ModelConfig.BATCH_SIZE, int)
        assert ModelConfig.BATCH_SIZE > 0
        assert isinstance(ModelConfig.LEARNING_RATE, float)
        assert ModelConfig.LEARNING_RATE > 0
    
    def test_constraint_weight(self):
        """Test constraint weight configuration."""
        assert isinstance(ModelConfig.CONSTRAINT_WEIGHT, float)
        assert ModelConfig.CONSTRAINT_WEIGHT > 0


class TestDataConfig:
    """Tests for DataConfig class."""
    
    def test_split_sizes(self):
        """Test train/test split configurations."""
        assert isinstance(DataConfig.TEST_SIZE, float)
        assert 0 < DataConfig.TEST_SIZE < 1
        assert isinstance(DataConfig.VAL_SIZE, float)
        assert 0 < DataConfig.VAL_SIZE < 1
    
    def test_random_state(self):
        """Test random state configuration."""
        assert isinstance(DataConfig.RANDOM_STATE, int)
    
    def test_composition_limits(self):
        """Test composition validation limits."""
        assert isinstance(DataConfig.MIN_TOTAL_COMPOSITION, float)
        assert isinstance(DataConfig.MAX_TOTAL_COMPOSITION, float)
        assert DataConfig.MIN_TOTAL_COMPOSITION < DataConfig.MAX_TOTAL_COMPOSITION
        assert isinstance(DataConfig.MAX_CARBON_CONTENT, float)
        assert DataConfig.MAX_CARBON_CONTENT > 0


class TestInverseDesignConfig:
    """Tests for InverseDesignConfig class."""
    
    def test_ga_parameters(self):
        """Test genetic algorithm parameters."""
        assert isinstance(InverseDesignConfig.POPULATION_SIZE, int)
        assert InverseDesignConfig.POPULATION_SIZE > 0
        assert isinstance(InverseDesignConfig.GENERATIONS, int)
        assert InverseDesignConfig.GENERATIONS > 0
    
    def test_mutation_and_crossover_rates(self):
        """Test mutation and crossover rates."""
        assert isinstance(InverseDesignConfig.MUTATION_RATE, float)
        assert 0 <= InverseDesignConfig.MUTATION_RATE <= 1
        assert isinstance(InverseDesignConfig.CROSSOVER_RATE, float)
        assert 0 <= InverseDesignConfig.CROSSOVER_RATE <= 1
    
    def test_element_ranges(self):
        """Test element range constraints."""
        assert isinstance(InverseDesignConfig.ELEMENT_RANGES, dict)
        assert len(InverseDesignConfig.ELEMENT_RANGES) == 13
        
        for elem, (min_val, max_val) in InverseDesignConfig.ELEMENT_RANGES.items():
            assert elem in ELEMENT_COLS
            assert isinstance(min_val, (int, float))
            assert isinstance(max_val, (int, float))
            assert min_val < max_val


class TestLoggingConfig:
    """Tests for LoggingConfig class."""
    
    def test_log_file_path(self):
        """Test log file path configuration."""
        assert isinstance(LoggingConfig.LOG_FILE, Path)
        assert LoggingConfig.LOG_FILE.parent == LOGS_DIR
    
    def test_log_levels(self):
        """Test log level configurations."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        assert LoggingConfig.LOG_LEVEL in valid_levels
        assert LoggingConfig.CONSOLE_LOG_LEVEL in valid_levels
        assert LoggingConfig.FILE_LOG_LEVEL in valid_levels
    
    def test_rotation_settings(self):
        """Test log rotation settings."""
        assert isinstance(LoggingConfig.MAX_BYTES, int)
        assert LoggingConfig.MAX_BYTES > 0
        assert isinstance(LoggingConfig.BACKUP_COUNT, int)
        assert LoggingConfig.BACKUP_COUNT >= 0


class TestAppConfig:
    """Tests for AppConfig class."""
    
    def test_page_settings(self):
        """Test Streamlit page settings."""
        assert isinstance(AppConfig.PAGE_TITLE, str)
        assert isinstance(AppConfig.PAGE_ICON, str)
        assert isinstance(AppConfig.LAYOUT, str)
        assert AppConfig.LAYOUT in ['centered', 'wide']
    
    def test_github_url(self):
        """Test GitHub URL configuration."""
        assert isinstance(AppConfig.GITHUB_URL, str)
        assert AppConfig.GITHUB_URL.startswith('https://github.com/')
    
    def test_presets(self):
        """Test preset compositions."""
        assert isinstance(AppConfig.PRESETS, dict)
        assert len(AppConfig.PRESETS) > 0
        
        for name, composition in AppConfig.PRESETS.items():
            assert isinstance(name, str)
            assert isinstance(composition, dict)
            # Check all elements are present
            for elem in ELEMENT_COLS:
                assert elem in composition
                assert isinstance(composition[elem], (int, float))
