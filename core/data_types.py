"""
Data structures and typing definitions for Autoresearch V2.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Detection:
    """Represents a single cell centroid detection in 3D space at timepoint t."""
    id: int
    t: int
    pos: np.ndarray  # [z, y, x] in voxel coordinates
    conf: float
    is_mitosis: float = 0.0
    features: Optional[np.ndarray] = None


@dataclass
class LineageEdge:
    """Directed edge from parent cell in frame t to child cell in frame t+1 or t+2."""
    source_id: int
    target_id: int
    cost: float = 0.0
    edge_prob: float = 1.0


@dataclass
class MitosisEvent:
    """Mitotic division event linking one parent to two daughter cells."""
    mother_id: int
    daughter1_id: int
    daughter2_id: int
    t_division: int
    confidence: float = 1.0


@dataclass
class DiagnosticTelemetry:
    """Rich diagnostic breakdown of tracking quality beyond scalar score."""
    total_score: float
    adjusted_edge_jaccard: float
    division_jaccard: float
    unadjusted_edge_jaccard: float
    
    # Node detection diagnostics
    node_recall_7um: float
    total_predicted_nodes: int
    estimated_true_nodes: int
    node_penalty_multiplier: float
    spurious_nodes: int
    missed_gt_nodes: int
    
    # Division diagnostics
    div_tp: int
    div_fp: int
    div_fn: int
    
    # Edge association diagnostics
    edge_tp: int
    edge_fp: int
    edge_fn: int
    edges_recovered: int
    edges_fragmented: int
    edges_lost_to_detection: int
    wrong_association_edges: int
    
    # Spatial errors
    z_axis_displacement_error: float = 0.0
    xy_displacement_error: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
