"""
Circuit visualization for sparse pruned models.

This module provides tools to visualize pruned circuits from MaskedSparseGPT models,
including:
- Circuit extraction from pruned model checkpoints
- Dashboard generation with top/bottom activating examples
- D3.js-based interactive HTML visualization

Example usage:
    from sparse_pretrain.src.visualization import visualize_circuit
    
    # Full pipeline from checkpoint to HTML
    graph, dashboards = visualize_circuit(
        checkpoint_path="outputs/pruning/dummy_quote",
        output_path="outputs/circuit_viewer.html",
    )
    
    # Or step by step:
    from sparse_pretrain.src.visualization import (
        extract_circuit,
        compute_dashboards,
        export_circuit_to_html,
    )
    
    graph = extract_circuit("outputs/pruning/dummy_quote")
    dashboards = compute_dashboards(graph, model, tokenizer)
    export_circuit_to_html(graph, "output.html", dashboards)
"""

from .circuit_extract import (
    CircuitGraph,
    CircuitNode,
    CircuitEdge,
    extract_circuit,
    save_circuit,
    load_circuit,
)

from .dashboard import (
    NodeDashboardData,
    ActivationExample,
    compute_dashboards,
    save_dashboards,
    load_dashboards,
)

from .circuit_viewer import (
    export_circuit_to_html,
    visualize_circuit,
)

__all__ = [
    # Circuit extraction
    "CircuitGraph",
    "CircuitNode", 
    "CircuitEdge",
    "extract_circuit",
    "save_circuit",
    "load_circuit",
    # Dashboard
    "NodeDashboardData",
    "ActivationExample",
    "compute_dashboards",
    "save_dashboards",
    "load_dashboards",
    # Viewer
    "export_circuit_to_html",
    "visualize_circuit",
]

