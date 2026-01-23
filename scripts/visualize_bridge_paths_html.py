#!/usr/bin/env python3
"""
Generate an HTML visualization of bridge path evaluation results.

Displays paths in a grid, colored by cross-entropy loss with a color scale.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class PathVisualization:
    """Parsed path information for visualization."""
    name: str
    cross_entropy: float
    kl_to_dense: float
    kl_to_sparse: float
    num_transitions: int
    # For each site: 'dense', 'sparse', or None
    model_at_site: List[str]
    # For each site: transition type if any ('encode', 'decode', 'both', or None)
    transition_at_site: List[str]


def parse_path_name(name: str, n_sites: int) -> Tuple[List[str], List[str]]:
    """
    Parse a path name to determine model at each site and transitions.
    
    Returns:
        model_at_site: List of 'dense' or 'sparse' for each site
        transition_at_site: List of transition type at each site ('encode', 'decode', 'both', None)
    """
    model_at_site = ['dense'] * n_sites  # Default to dense
    transition_at_site = [None] * n_sites
    
    if name == 'pure_dense':
        return model_at_site, transition_at_site
    
    if name == 'pure_sparse':
        return ['sparse'] * n_sites, transition_at_site
    
    # Parse path segments
    # Examples: s0->d, d1->s, d0->s2->d, s0->d4->s, d0->enc->dec->d, s4->dec->enc->s
    
    parts = name.split('->')
    
    if len(parts) == 2:
        # Single transition: s0->d, d1->s
        start = parts[0]
        end = parts[1]
        
        if start.startswith('s'):
            # Starts on sparse
            site = int(start[1:])
            # Sparse until site, then decode to dense
            for i in range(site + 1):
                model_at_site[i] = 'sparse'
            transition_at_site[site] = 'decode'
            for i in range(site + 1, n_sites):
                model_at_site[i] = 'dense'
        else:
            # Starts on dense (d0->s)
            site = int(start[1:])
            # Dense until site, then encode to sparse
            for i in range(site + 1):
                model_at_site[i] = 'dense'
            transition_at_site[site] = 'encode'
            for i in range(site + 1, n_sites):
                model_at_site[i] = 'sparse'
    
    elif len(parts) == 3:
        # Double transition or enc->dec
        start = parts[0]
        mid = parts[1]
        end = parts[2]
        
        if mid == 'enc' or mid == 'dec':
            # Double-back: d0->enc->dec->d or s4->dec->enc->s
            if start.startswith('d'):
                site = int(start[1:])
                # Dense throughout, but with encode+decode at site
                transition_at_site[site] = 'both'
            else:
                site = int(start[1:])
                model_at_site = ['sparse'] * n_sites
                transition_at_site[site] = 'both'
        else:
            # Two transitions: d0->s2->d, s0->d4->s
            if start.startswith('d'):
                site1 = int(start[1:])
                site2 = int(mid[1:])
                # Dense until site1, sparse until site2, then dense
                for i in range(site1 + 1):
                    model_at_site[i] = 'dense'
                transition_at_site[site1] = 'encode'
                for i in range(site1 + 1, site2 + 1):
                    model_at_site[i] = 'sparse'
                transition_at_site[site2] = 'decode'
                for i in range(site2 + 1, n_sites):
                    model_at_site[i] = 'dense'
            else:
                site1 = int(start[1:])
                site2 = int(mid[1:])
                # Sparse until site1, dense until site2, then sparse
                for i in range(site1 + 1):
                    model_at_site[i] = 'sparse'
                transition_at_site[site1] = 'decode'
                for i in range(site1 + 1, site2 + 1):
                    model_at_site[i] = 'dense'
                transition_at_site[site2] = 'encode'
                for i in range(site2 + 1, n_sites):
                    model_at_site[i] = 'sparse'
    
    elif len(parts) == 4:
        # d0->enc->dec->d style (already handled above, but 4 parts)
        start = parts[0]
        if start.startswith('d'):
            site = int(start[1:])
            transition_at_site[site] = 'both'
        else:
            site = int(start[1:])
            model_at_site = ['sparse'] * n_sites
            transition_at_site[site] = 'both'
    
    return model_at_site, transition_at_site


def load_results(json_path: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def parse_all_paths(results: Dict[str, Any]) -> List[PathVisualization]:
    """Parse all paths from results."""
    n_sites = results['num_sites']
    paths = []
    
    for name, data in results['paths'].items():
        model_at_site, transition_at_site = parse_path_name(name, n_sites)
        paths.append(PathVisualization(
            name=name,
            cross_entropy=data['cross_entropy'],
            kl_to_dense=data['kl_to_dense'],
            kl_to_sparse=data['kl_to_sparse'],
            num_transitions=data['num_transitions'],
            model_at_site=model_at_site,
            transition_at_site=transition_at_site,
        ))
    
    return paths


def ce_to_color(ce: float, min_ce: float, max_ce: float) -> str:
    """Convert CE to a red color (low CE = light, high CE = dark red)."""
    # Normalize to 0-1
    t = (ce - min_ce) / (max_ce - min_ce) if max_ce > min_ce else 0
    
    # Light red (best) to dark red (worst)
    # Light: rgb(255, 230, 230) -> Dark: rgb(180, 30, 30)
    r = int(255 - t * 75)
    g = int(230 - t * 200)
    b = int(230 - t * 200)
    
    return f'rgb({r}, {g}, {b})'


def generate_html(results: Dict[str, Any], paths: List[PathVisualization]) -> str:
    """Generate HTML visualization."""
    n_sites = results['num_sites']
    checkpoint = results['checkpoint'].split('/')[-1]
    
    # Sort paths by CE
    paths_sorted = sorted(paths, key=lambda p: p.cross_entropy)
    
    # Get min/max CE for color scale
    min_ce = min(p.cross_entropy for p in paths)
    max_ce = max(p.cross_entropy for p in paths)
    
    # Group by num_transitions
    pure_paths = [p for p in paths_sorted if p.num_transitions == 0]
    single_paths = [p for p in paths_sorted if p.num_transitions == 1]
    double_paths = [p for p in paths_sorted if p.num_transitions == 2]
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Bridge Path Evaluation - {checkpoint}</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }}
        h1 {{
            text-align: center;
            color: #ff6b6b;
            margin-bottom: 5px;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 10px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section-title {{
            font-size: 1.4em;
            color: #ffa07a;
            margin-bottom: 15px;
            border-bottom: 2px solid #ffa07a;
            padding-bottom: 5px;
        }}
        .paths-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: flex-start;
        }}
        .path-card {{
            background: #16213e;
            border-radius: 6px;
            padding: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        .path-metrics {{
            font-size: 0.7em;
            text-align: center;
            margin-bottom: 6px;
            color: #333;
            font-weight: 500;
        }}
        .ladder {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 1px;
        }}
        .ladder-row {{
            display: flex;
            align-items: center;
            height: 18px;
        }}
        .ladder-cell {{
            width: 18px;
            height: 18px;
            border-radius: 3px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 9px;
            font-weight: bold;
        }}
        .ladder-cell.dense {{
            background: #4a90d9;
            color: white;
        }}
        .ladder-cell.sparse {{
            background: #e67e22;
            color: white;
        }}
        .ladder-cell.inactive {{
            background: transparent;
        }}
        .ladder-connector {{
            width: 24px;
            height: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .connector-arrow {{
            font-size: 11px;
            color: #333;
        }}
        .colorbar-container {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: #16213e;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        .colorbar-title {{
            font-size: 0.8em;
            text-align: center;
            margin-bottom: 10px;
            color: #ddd;
        }}
        .colorbar {{
            width: 30px;
            height: 200px;
            background: linear-gradient(to bottom, rgb(180, 30, 30), rgb(255, 230, 230));
            border-radius: 4px;
            margin: 0 auto;
        }}
        .colorbar-labels {{
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 200px;
            margin-left: 10px;
            font-size: 0.7em;
            color: #aaa;
        }}
        .colorbar-row {{
            display: flex;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 15px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-box {{
            width: 18px;
            height: 18px;
            border-radius: 3px;
        }}
        .legend-box.dense {{
            background: #4a90d9;
        }}
        .legend-box.sparse {{
            background: #e67e22;
        }}
        .metrics-key {{
            text-align: center;
            color: #aaa;
            font-size: 0.85em;
            margin-bottom: 25px;
        }}
        .metrics-key code {{
            background: #16213e;
            padding: 2px 6px;
            border-radius: 3px;
            color: #ddd;
        }}
    </style>
</head>
<body>
    <h1>Bridge Path Evaluation</h1>
    <p class="subtitle">{checkpoint} · {results['num_tokens']:,} tokens · {len(paths)} paths</p>
    
    <div class="legend">
        <div class="legend-item">
            <div class="legend-box dense"></div>
            <span>Dense</span>
        </div>
        <div class="legend-item">
            <div class="legend-box sparse"></div>
            <span>Sparse</span>
        </div>
    </div>
    
    <div class="metrics-key">
        Values shown: <code>CE | KL→Dense | KL→Sparse</code>
    </div>
    
    <div class="colorbar-container">
        <div class="colorbar-title">Cross-Entropy</div>
        <div class="colorbar-row">
            <div class="colorbar"></div>
            <div class="colorbar-labels">
                <span>{max_ce:.2f}</span>
                <span>{(max_ce + min_ce) / 2:.2f}</span>
                <span>{min_ce:.2f}</span>
            </div>
        </div>
    </div>
'''
    
    def render_path_card(path: PathVisualization) -> str:
        """Render a single path card."""
        bg_color = ce_to_color(path.cross_entropy, min_ce, max_ce)
        
        card_html = f'''
        <div class="path-card" style="background: {bg_color};">
            <div class="path-metrics">
                {path.cross_entropy:.2f} | {path.kl_to_dense:.2f} | {path.kl_to_sparse:.2f}
            </div>
            <div class="ladder">
'''
        
        # Render sites bottom to top (reverse order)
        for site in reversed(range(n_sites)):
            model = path.model_at_site[site]
            trans = path.transition_at_site[site]
            
            dense_class = 'dense' if model == 'dense' else 'inactive'
            sparse_class = 'sparse' if model == 'sparse' else 'inactive'
            
            # Only show arrows where bridges are traversed
            if trans == 'encode':
                connector = '<span class="connector-arrow">→</span>'
            elif trans == 'decode':
                connector = '<span class="connector-arrow">←</span>'
            elif trans == 'both':
                connector = '<span class="connector-arrow">⇄</span>'
            else:
                connector = ''
            
            card_html += f'''
                <div class="ladder-row">
                    <div class="ladder-cell {dense_class}">{"D" if model == "dense" else ""}</div>
                    <div class="ladder-connector">{connector}</div>
                    <div class="ladder-cell {sparse_class}">{"S" if model == "sparse" else ""}</div>
                </div>
'''
        
        card_html += '''
            </div>
        </div>
'''
        return card_html
    
    def render_section(title: str, section_paths: List[PathVisualization]) -> str:
        """Render a section of paths."""
        if not section_paths:
            return ''
        
        section_html = f'''
    <div class="section">
        <div class="section-title">{title} ({len(section_paths)} paths)</div>
        <div class="paths-grid">
'''
        for path in section_paths:
            section_html += render_path_card(path)
        
        section_html += '''
        </div>
    </div>
'''
        return section_html
    
    html += render_section('Pure Models (0 transitions)', pure_paths)
    html += render_section('Single Transition', single_paths)
    html += render_section('Double Transition', double_paths)
    
    html += '''
</body>
</html>
'''
    return html


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML visualization of bridge path evaluation results"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to path_evaluation.json file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output HTML file path (default: same as input with .html extension)"
    )
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.input)
    
    # Parse paths
    paths = parse_all_paths(results)
    
    # Generate HTML
    html = generate_html(results, paths)
    
    # Save
    output_path = args.output or str(Path(args.input).with_suffix('.html'))
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"HTML visualization saved to: {output_path}")


if __name__ == "__main__":
    main()

