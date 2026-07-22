from dataclasses import dataclass, field
from typing import Dict, Tuple

@dataclass
class Vis2DConfig:
    # =========================================================================
    # 3D Plotting Options (Most commonly adjusted settings)
    # =========================================================================

    # Scale factor for the 3D bounding box range
    # Increase (e.g., 1.2, 2.0) -> Trajectory appears smaller, more white space around it
    # Decrease (e.g., 0.5) -> Trajectory appears larger, hugging the borders tightly
    plot_3d_range_scale: float = 1.0
    
    # Scale factor for the length of Interaction Frame arrows
    # Default is 0.02. If arrows are too short, increase to 0.05 or 0.1
    plot_3d_arrow_scale: float = 0.02


    # =========================================================================
    # 2D Plotting Options (Fine-tuning parameters)
    # =========================================================================

    # Font sizes for various plot elements
    rc_params: Dict[str, float] = field(default_factory=lambda: {
        'font.size': 13,
        'axes.titlesize': 16,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 13,
        'legend.title_fontsize': 15
    })
    
    # Figure dimensions (width, height)
    figsize: Tuple[int, int] = (20, 12)
    
    # Velocity Line settings
    vel_color: str = 'tab:blue'
    vel_linewidth: float = 1.0
    vel_linestyle: str = '-'
    
    # Force Line settings
    force_color: str = 'tab:orange'
    force_linewidth: float = 1.0
    force_linestyle: str = '--'
    
    # Reference target Line settings
    ref_color: str = 'k'
    ref_linewidth: float = 1.5
    ref_linestyle: str = ':'
    ref_alpha: float = 0.7
    
    # Task ID background spanning sections settings
    bg_alpha: float = 0.15
    bg_linewidth: float = 0.0
    
    # Subplot title padding and size
    title_pad: float = 2.0
    title_fontsize: int = 16
