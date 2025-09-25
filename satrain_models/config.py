"""Configuration classes for SatRain dataset and models."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
import sys
from pathlib import Path

# Try to import TOML library (tomllib in Python 3.11+, tomli for older versions)
try:
    import tomllib
    TOML_AVAILABLE = True
except ImportError:
    try:
        import tomli as tomllib
        TOML_AVAILABLE = True
    except ImportError:
        TOML_AVAILABLE = False
        tomllib = None

# Try to import from satrain package if available
try:
    from satrain.input import InputConfig, parse_retrieval_inputs
    from satrain.target import TargetConfig
    SATRAIN_AVAILABLE = True
except ImportError:
    SATRAIN_AVAILABLE = False
    InputConfig = None
    TargetConfig = None
    parse_retrieval_inputs = None


class SatRainConfig:
    """Configuration for SatRain dataset including geometry, subset, retrieval input and target config.
    
    This class provides automatic parsing of retrieval inputs and target configurations
    from dictionaries when the satrain package is available.
    """
    
    geometry: Optional[str] = None  # "gridded" or "on_swath"
    subset: Optional[str] = None    # "xs", "s", "m", "l", "xl"
    retrieval_input: Optional[List[Union[str, Dict[str, Any]]]] = None
    target_config: Optional[Union[Dict[str, Any]]] = None
    
    def __init__(self, **kwargs):
        """Initialize with flexible keyword arguments from TOML."""
        self.geometry = kwargs.get('geometry')
        self.subset = kwargs.get('subset') 
        self.retrieval_input = kwargs.get('retrieval_input')
        self.target_config = kwargs.get('target_config')
        
        # Store other keys as attributes for flexibility
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)
                
        # Call post-init processing
        self.__post_init__()
    
    def __post_init__(self):
        """Parse retrieval_input and target_config after initialization."""
        if SATRAIN_AVAILABLE:
            # Parse retrieval inputs if provided
            if self.retrieval_input is not None:
                self.retrieval_input = parse_retrieval_inputs(self.retrieval_input)
            
            # Parse target config if provided as dict
            if isinstance(self.target_config, dict):
                self.target_config = TargetConfig(**self.target_config)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SatRainConfig":
        """Create SatRainConfig from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            SatRainConfig instance
        """
        return cls(**config_dict)
    
    @classmethod
    def from_toml_file(cls, toml_path: Union[str, Path]) -> "SatRainConfig":
        """Create SatRainConfig from TOML file.
        
        Args:
            toml_path: Path to the TOML configuration file
            
        Returns:
            SatRainConfig instance
            
        Raises:
            ImportError: If TOML library is not available
            FileNotFoundError: If the TOML file doesn't exist
        """
        if not TOML_AVAILABLE:
            raise ImportError(
                "TOML library not available. Install with: pip install tomli"
            )
        
        toml_path = Path(toml_path)
        if not toml_path.exists():
            raise FileNotFoundError(f"TOML file not found: {toml_path}")
        
        with open(toml_path, "rb") as f:
            config_dict = tomllib.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_toml_string(cls, toml_string: str) -> "SatRainConfig":
        """Create SatRainConfig from TOML string.
        
        Args:
            toml_string: TOML configuration as string
            
        Returns:
            SatRainConfig instance
            
        Raises:
            ImportError: If TOML library is not available
        """
        if not TOML_AVAILABLE:
            raise ImportError(
                "TOML library not available. Install with: pip install tomli"
            )
        
        config_dict = tomllib.loads(toml_string)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert SatRainConfig to dictionary representation.
        
        Returns:
            Dictionary representation of the configuration
        """
        result = {}
        
        if self.geometry is not None:
            result["geometry"] = self.geometry
        if self.subset is not None:
            result["subset"] = self.subset
        if self.retrieval_input is not None:
            if SATRAIN_AVAILABLE and self.retrieval_input and hasattr(self.retrieval_input[0], 'to_dict'):
                result["retrieval_input"] = [inp.to_dict() for inp in self.retrieval_input]
            else:
                result["retrieval_input"] = self.retrieval_input
        if self.target_config is not None:
            if SATRAIN_AVAILABLE and hasattr(self.target_config, 'to_dict'):
                result["target_config"] = self.target_config.to_dict()
            else:
                result["target_config"] = self.target_config
                
        return result
    
    def to_toml_file(self, toml_path: Union[str, Path]) -> None:
        """Write SatRainConfig to TOML file.
        
        Args:
            toml_path: Path where to write the TOML configuration file
            
        Raises:
            ImportError: If TOML writing library is not available
        """
        try:
            import tomli_w
        except ImportError:
            raise ImportError(
                "TOML writing library not available. Install with: pip install tomli-w"
            )
        
        toml_path = Path(toml_path)
        config_dict = self.to_dict()
        
        with open(toml_path, "wb") as f:
            tomli_w.dump(config_dict, f)