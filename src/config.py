"""
HEPAR Configuration Loader
==========================
Load YAML configs for reproducible experiments.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class QuadtreeConfig:
    threshold: float = 0.02
    min_block: int = 1
    max_depth: Optional[int] = None


@dataclass
class EncodingConfig:
    use_spae: bool = True
    use_phase: bool = True
    intensity_bits: int = 1


@dataclass
class CompilationConfig:
    gray_code: bool = True
    topology_aware: bool = True
    optimization_level: int = 2


@dataclass
class MitigationConfig:
    trex: bool = True
    dd: bool = False
    calibration_shots: int = 4096


@dataclass
class NoiseConfig:
    enabled: bool = True
    p1_error: float = 0.001
    p2_error: float = 0.01
    read_error: float = 0.02


@dataclass
class AblationConfig:
    """HEPAR-minus-X experiment toggles."""
    hepar_no_qt: bool = False    # Disable quadtree
    hepar_no_gc: bool = False    # Disable Gray code
    hepar_no_spae: bool = False  # Disable phase encoding
    hepar_no_trex: bool = False  # Disable TREX
    hepar_no_dd: bool = False    # Disable DD


@dataclass
class HEPARConfig:
    """Complete HEPAR configuration."""
    quadtree: QuadtreeConfig = field(default_factory=QuadtreeConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    compilation: CompilationConfig = field(default_factory=CompilationConfig)
    mitigation: MitigationConfig = field(default_factory=MitigationConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    shots: int = 8192
    seed: int = 42
    datasets: List[str] = field(default_factory=lambda: ["Lena", "MRI", "Noise"])
    image_size: int = 32
    simulate_baselines: bool = False # Skip heavy NEQR/FRQI simulations


def load_config(path: str = "configs/hepar.yaml") -> HEPARConfig:
    """Load configuration from YAML file."""
    if not os.path.exists(path):
        print(f"Config not found at {path}, using defaults")
        return HEPARConfig()
    
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    config = HEPARConfig()
    
    # Parse HEPAR section
    if 'hepar' in data:
        h = data['hepar']
        if 'quadtree' in h:
            config.quadtree = QuadtreeConfig(**h['quadtree'])
        if 'encoding' in h:
            config.encoding = EncodingConfig(**h['encoding'])
        if 'compilation' in h:
            config.compilation = CompilationConfig(**h['compilation'])
        if 'mitigation' in h:
            config.mitigation = MitigationConfig(**h['mitigation'])
    
    # Parse other sections
    if 'simulation' in data:
        config.shots = data['simulation'].get('shots', 8192)
        config.seed = data['simulation'].get('seed', 42)
    
    if 'noise' in data:
        config.noise = NoiseConfig(**data['noise'])
    
    if 'datasets' in data:
        config.datasets = data['datasets'].get('names', ["Lena", "MRI", "Noise"])
        config.image_size = data['datasets'].get('size', 32)
    
    if 'ablation' in data:
        config.ablation = AblationConfig(**data['ablation'])
    
    return config


def get_ablation_variants() -> Dict[str, dict]:
    """
    Return configs for HEPAR-minus-X experiments.
    
    Each variant disables one pillar to prove its necessity.
    """
    base = HEPARConfig()
    
    return {
        'HEPAR-full': {},
        'HEPAR-no-QT': {'ablation': {'hepar_no_qt': True}},
        'HEPAR-no-GC': {'ablation': {'hepar_no_gc': True}},
        'HEPAR-no-SPAE': {'ablation': {'hepar_no_spae': True}},
        'HEPAR-no-TREX': {'ablation': {'hepar_no_trex': True}},
        'HEPAR-no-DD': {'ablation': {'hepar_no_dd': True}},
    }


if __name__ == "__main__":
    config = load_config()
    print(f"Loaded config: shots={config.shots}, datasets={config.datasets}")
    print(f"TREX enabled: {config.mitigation.trex}")
    print(f"Gray code: {config.compilation.gray_code}")
