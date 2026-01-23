"""
CLI entry point for visualizing pruned circuits.

Usage:
    python -m my_sparse_pretrain.src.visualization.run_viewer \
        --checkpoint_path outputs/pruning/dummy_quote \
        --output_path outputs/circuit_viewer.html

Or with custom options:
    python -m my_sparse_pretrain.src.visualization.run_viewer \
        --checkpoint_path outputs/pruning/dummy_quote \
        --output_path outputs/circuit_viewer.html \
        --data_source task \
        --n_samples 1000 \
        --edge_threshold 0.01
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a pruned circuit from a MaskedSparseGPT checkpoint"
    )
    
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to pruning output directory (containing binary_masks.pt and results.json)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output HTML file path (defaults to checkpoint_path/circuit_viewer.html)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Override model path (uses checkpoint's model_path if not specified)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name (defaults to model's tokenizer)",
    )
    
    # Dashboard options
    parser.add_argument(
        "--no_dashboards",
        action="store_true",
        help="Skip dashboard computation (faster, but no activation examples)",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="simplestories",
        choices=["simplestories", "task"],
        help="Data source for dashboard computation",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=500,
        help="Number of samples for dashboard computation",
    )
    parser.add_argument(
        "--k_top",
        type=int,
        default=10,
        help="Number of top activating examples per node",
    )
    parser.add_argument(
        "--k_bottom",
        type=int,
        default=10,
        help="Number of bottom activating examples per node",
    )
    
    # Graph options
    parser.add_argument(
        "--edge_threshold",
        type=float,
        default=0.0,
        help="Minimum absolute edge weight to include",
    )
    parser.add_argument(
        "--no_resid_nodes",
        action="store_true",
        help="Exclude residual stream nodes from the graph",
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for computation",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    
    args = parser.parse_args()
    
    # Set output path
    checkpoint_path = Path(args.checkpoint_path)
    if args.output_path is None:
        output_path = checkpoint_path / "circuit_viewer.html"
    else:
        output_path = Path(args.output_path)
    
    verbose = not args.quiet
    
    # Import here to avoid slow imports when just showing help
    from .circuit_viewer import visualize_circuit
    
    # Run visualization
    graph, dashboards = visualize_circuit(
        checkpoint_path=str(checkpoint_path),
        output_path=str(output_path),
        model_path=args.model_path,
        tokenizer_name=args.tokenizer,
        compute_dashboards_flag=not args.no_dashboards,
        data_source=args.data_source,
        n_samples=args.n_samples,
        edge_threshold=args.edge_threshold,
        device=args.device,
        verbose=verbose,
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print("CIRCUIT VISUALIZATION COMPLETE")
        print(f"{'='*60}")
        summary = graph.summary()
        print(f"Nodes: {summary['num_nodes']}")
        print(f"Edges: {summary['num_edges']}")
        print(f"Layers: {summary['n_layers']}")
        if dashboards:
            print(f"Dashboard nodes: {len(dashboards)}")
        print(f"\nOutput: {output_path}")
        print(f"Open in browser to view the circuit graph.")


if __name__ == "__main__":
    main()

