"""
Demo script showing how to use the circuit visualization module.

This demonstrates the full pipeline from pruned model checkpoint to
interactive HTML visualization.

Run from the global_circuits directory:
    python sparse_pretrain/notebooks/circuit_visualization_demo.py
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def demo_full_pipeline():
    """
    Demo: Full visualization pipeline from checkpoint.
    
    Assumes you have already run pruning and have a checkpoint at:
    sparse_pretrain/outputs/pruning/<task_name>/
    """
    from sparse_pretrain.src.visualization import visualize_circuit
    
    # Set your checkpoint path here
    checkpoint_path = "sparse_pretrain/outputs/pruning/dummy_quote"
    output_path = "sparse_pretrain/outputs/circuit_viewer.html"
    
    print("Running full visualization pipeline...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Output: {output_path}")
    
    graph, dashboards = visualize_circuit(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        compute_dashboards_flag=True,
        data_source="simplestories",
        n_samples=500,
        device="cuda",
        verbose=True,
    )
    
    print(f"\nDone! Open {output_path} in a browser to view the circuit.")
    return graph, dashboards


def demo_step_by_step():
    """
    Demo: Step-by-step visualization with more control.
    """
    import torch
    from transformers import AutoTokenizer
    
    from sparse_pretrain.src.visualization import (
        extract_circuit,
        compute_dashboards,
        export_circuit_to_html,
        save_circuit,
        save_dashboards,
    )
    from sparse_pretrain.src.pruning.run_pruning import load_model
    
    # Set paths
    checkpoint_path = "sparse_pretrain/outputs/pruning/dummy_quote"
    device = "cuda"
    
    # Step 1: Extract circuit graph
    print("Step 1: Extracting circuit graph...")
    graph = extract_circuit(
        checkpoint_path=checkpoint_path,
        edge_threshold=0.0,  # Include all edges
        include_resid_nodes=True,
        device=device,
    )
    
    summary = graph.summary()
    print(f"  Nodes: {summary['num_nodes']}")
    print(f"  Edges: {summary['num_edges']}")
    
    # Optionally save the graph
    save_circuit(graph, "sparse_pretrain/outputs/circuit_graph.json")
    print("  Saved graph to circuit_graph.json")
    
    # Step 2: Load model and tokenizer for dashboards
    print("\nStep 2: Loading model for dashboard computation...")
    
    import json
    with open(f"{checkpoint_path}/results.json") as f:
        results = json.load(f)
    
    model_path = results["model_path"]
    model, config_dict = load_model(model_path, device=device)
    
    tokenizer_name = config_dict.get("training_config", {}).get("tokenizer_name")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Step 3: Compute dashboards
    print("\nStep 3: Computing dashboards...")
    dashboards = compute_dashboards(
        graph=graph,
        model=model,
        tokenizer=tokenizer,
        data_source="simplestories",  # or "task" if you want task-specific examples
        n_samples=500,
        k_top=10,
        k_bottom=10,
        device=device,
        verbose=True,
    )
    
    print(f"  Computed dashboards for {len(dashboards)} nodes")
    
    # Optionally save dashboards
    save_dashboards(dashboards, "sparse_pretrain/outputs/dashboards.json")
    print("  Saved dashboards to dashboards.json")
    
    # Step 4: Export to HTML
    print("\nStep 4: Exporting to HTML...")
    output_path = "sparse_pretrain/outputs/circuit_viewer_stepwise.html"
    export_circuit_to_html(
        graph=graph,
        output_path=output_path,
        dashboards=dashboards,
        verbose=True,
    )
    
    print(f"\nDone! Open {output_path} in a browser.")
    return graph, dashboards


def demo_with_task_data():
    """
    Demo: Use task-specific data for dashboards instead of SimpleStories.
    """
    from transformers import AutoTokenizer
    
    from sparse_pretrain.src.visualization import (
        extract_circuit,
        compute_dashboards,
        export_circuit_to_html,
    )
    from sparse_pretrain.src.pruning.run_pruning import load_model
    from sparse_pretrain.src.pruning.tasks import get_task, TaskDataset
    
    checkpoint_path = "sparse_pretrain/outputs/pruning/dummy_quote"
    device = "cuda"
    
    print("Extracting circuit...")
    graph = extract_circuit(checkpoint_path, device=device)
    
    print("Loading model...")
    import json
    with open(f"{checkpoint_path}/results.json") as f:
        results = json.load(f)
    
    model_path = results["model_path"]
    task_name = results["task"]
    model, config_dict = load_model(model_path, device=device)
    
    tokenizer_name = config_dict.get("training_config", {}).get("tokenizer_name")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create task for task-specific dashboard data
    task = get_task(task_name, tokenizer)
    
    print(f"Computing dashboards using task '{task_name}' data...")
    dashboards = compute_dashboards(
        graph=graph,
        model=model,
        tokenizer=tokenizer,
        data_source="task",
        task=task,
        n_samples=500,
        device=device,
        verbose=True,
    )
    
    output_path = "sparse_pretrain/outputs/circuit_viewer_task.html"
    export_circuit_to_html(graph, output_path, dashboards)
    print(f"\nDone! Open {output_path} in a browser.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["full", "step", "task"],
        default="full",
        help="Demo mode: full pipeline, step-by-step, or with task data"
    )
    args = parser.parse_args()
    
    if args.mode == "full":
        demo_full_pipeline()
    elif args.mode == "step":
        demo_step_by_step()
    elif args.mode == "task":
        demo_with_task_data()

