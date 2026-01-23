#!/usr/bin/env python3
"""
Update all existing circuit HTML files with the new template and copy to a central location.
"""
import re
import json
import sys
import shutil
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sparse_pretrain.scripts.run_single_pruning import _get_lightweight_html_template


def extract_data_from_html(html_content: str) -> tuple:
    """Extract graphData and dashboardData from existing HTML."""
    # Extract graphData (contains nodes, edges, metadata)
    graph_match = re.search(r'const graphData = ({.*?});', html_content, re.DOTALL)
    if not graph_match:
        raise ValueError("Could not find graphData in HTML")
    graph_data = json.loads(graph_match.group(1))
    
    # Extract dashboardData
    dash_match = re.search(r'const dashboardData = ({.*?});', html_content, re.DOTALL)
    if not dash_match:
        raise ValueError("Could not find dashboardData in HTML")
    dashboard_data = json.loads(dash_match.group(1))
    
    return graph_data, dashboard_data


def get_informative_name(model_dir: str, html_name: str) -> str:
    """Generate an informative filename from the model directory and html name."""
    # Parse the model directory name
    # Examples: ss_bridges_d1024_f0.015625_mean, ss_d128_f1_zero_noembed
    
    parts = model_dir.split('_')
    
    # Extract key info
    is_bridges = 'bridges' in model_dir
    is_noembed = 'noembed' in model_dir
    
    # Find d_model
    d_model = None
    for p in parts:
        if p.startswith('d') and p[1:].isdigit():
            d_model = p[1:]
            break
    
    # Find ablation type (mean or zero)
    ablation = 'mean' if 'mean' in model_dir else 'zero'
    
    # Build name
    model_type = 'bridges' if is_bridges else 'base'
    embed_type = 'noembed' if is_noembed else 'embed'
    
    return f"circuit_{model_type}_d{d_model}_{ablation}_{embed_type}.html"


def update_html_file(html_path: Path) -> tuple:
    """Update an HTML file with the new template. Returns (success, new_content)."""
    with open(html_path, 'r') as f:
        old_html = f.read()
    
    try:
        graph_data, dashboard_data = extract_data_from_html(old_html)
    except ValueError as e:
        return False, str(e)
    
    # Generate new HTML with updated template
    html = _get_lightweight_html_template()
    html = html.replace("__GRAPH_DATA__", json.dumps(graph_data))
    html = html.replace("__DASHBOARD_DATA__", json.dumps(dashboard_data))
    
    return True, html


def main():
    base = Path("outputs/carbs_results_pronoun")
    output_dir = base / "all_circuits"
    output_dir.mkdir(exist_ok=True)
    
    # Find all model directories (ss_* directories)
    model_dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith('ss_')]
    
    print(f"Found {len(model_dirs)} model directories\n")
    
    results = []
    
    for model_dir in sorted(model_dirs):
        model_name = model_dir.name
        
        # Find the circuit HTML for this model
        # Check root first, then best_checkpoint/evals/
        html_path = model_dir / "circuit.html"
        if not html_path.exists():
            # Look in best_checkpoint/evals/
            evals_dir = model_dir / "best_checkpoint" / "evals"
            if evals_dir.exists():
                circuit_htmls = list(evals_dir.glob("circuit_*.html"))
                if circuit_htmls:
                    html_path = circuit_htmls[0]  # Take the first one
        
        if not html_path.exists():
            print(f"Skipping {model_name}: no circuit HTML found")
            continue
        
        # Get informative name
        new_name = get_informative_name(model_name, html_path.name)
        
        print(f"Processing: {model_name}")
        print(f"  Source: {html_path.relative_to(base)}")
        print(f"  Output: {new_name}")
        
        # Update the HTML
        success, content = update_html_file(html_path)
        
        if not success:
            print(f"  ERROR: {content}")
            continue
        
        # Write to output directory
        output_path = output_dir / new_name
        with open(output_path, 'w') as f:
            f.write(content)
        
        results.append((model_name, new_name))
        print(f"  ✓ Saved")
    
    print(f"\n{'='*60}")
    print(f"Generated {len(results)} circuit HTMLs in: {output_dir}")
    print("\nFiles:")
    for model_name, name in sorted(results, key=lambda x: x[1]):
        print(f"  {name}")


if __name__ == "__main__":
    main()

