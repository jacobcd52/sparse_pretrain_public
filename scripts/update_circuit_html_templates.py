#!/usr/bin/env python3
"""
Quick script to update existing circuit HTML files with the new template.
This extracts the graph data and dashboards from existing HTMLs and re-wraps them
with the updated template from run_single_pruning.py.

Much faster than regenerating from scratch since no model loading or computation needed.
"""
import re
import json
import sys
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


def update_html_with_new_template(html_path: Path) -> None:
    """Update an HTML file with the new template."""
    print(f"Updating: {html_path.parent.name}")
    
    # Read existing HTML
    with open(html_path, 'r') as f:
        old_html = f.read()
    
    # Extract data
    try:
        graph_data, dashboard_data = extract_data_from_html(old_html)
    except ValueError as e:
        print(f"  ERROR: {e}")
        return
    
    # Generate new HTML with updated template
    html = _get_lightweight_html_template()
    html = html.replace("__GRAPH_DATA__", json.dumps(graph_data))
    html = html.replace("__DASHBOARD_DATA__", json.dumps(dashboard_data))
    
    # Write updated HTML
    with open(html_path, 'w') as f:
        f.write(html)
    
    print(f"  Updated successfully")


def main():
    # Find all noembed circuit.html files
    base = Path("outputs/carbs_results_pronoun")
    noembed_htmls = sorted(base.glob("*noembed*/circuit.html"))
    
    print(f"Found {len(noembed_htmls)} circuit.html files to update\n")
    
    for html_path in noembed_htmls:
        update_html_with_new_template(html_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

