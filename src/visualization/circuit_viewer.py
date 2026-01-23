"""
Circuit Viewer - Static HTML export for sparse circuit graphs.

Generates a self-contained HTML file that visualizes pruned circuit graphs
using D3.js, with dashboards displayed on hover showing top/bottom activations.
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .circuit_extract import CircuitGraph, CircuitNode, CircuitEdge
from .dashboard import NodeDashboardData


def _generate_dashboard_html(
    dashboard: NodeDashboardData,
    window_size: int = 24,
) -> str:
    """
    Generate compact HTML for a single node's dashboard.
    Shows both top (positive) and bottom (negative) activating examples.
    Includes windowed and full context views with an Expand All button.
    """
    html_parts = []
    
    # Header with statistics and Expand All button
    html_parts.append(
        f'<div style="margin-bottom:10px;padding:6px 8px;background:#1a1a2e;border-radius:4px;display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="color:#888;font-size:10px;">mean: {dashboard.mean_activation:.3f} | '
        f'std: {dashboard.std_activation:.3f} | freq: {dashboard.frequency*100:.2f}%</span>'
        f'<button class="expand-all-btn" onclick="toggleExpandAll(this)" style="background:#2a3f5f;color:#fff;border:1px solid #4a90d9;border-radius:4px;padding:4px 10px;font-size:10px;cursor:pointer;">Expand All</button>'
        f'</div>'
    )
    
    # Compute global max and min absolute values across ALL examples (both top and bottom)
    # This ensures consistent color scaling across the entire dashboard
    # Color intensity will be scaled to the range [min_abs, max_abs]
    global_max_abs = 0.0
    global_min_abs = float('inf')
    for ex in dashboard.top_examples[:10]:
        if ex.activations:
            for a in ex.activations:
                if not math.isnan(a) and not math.isinf(a):
                    abs_a = abs(a)
                    global_max_abs = max(global_max_abs, abs_a)
                    global_min_abs = min(global_min_abs, abs_a)
    for ex in dashboard.bottom_examples[:10]:
        if ex.activations:
            for a in ex.activations:
                if not math.isnan(a) and not math.isinf(a):
                    abs_a = abs(a)
                    global_max_abs = max(global_max_abs, abs_a)
                    global_min_abs = min(global_min_abs, abs_a)
    
    # Ensure we have valid values
    if global_max_abs == 0.0:
        global_max_abs = 1e-6
    if global_min_abs == float('inf'):
        global_min_abs = 0.0
    
    # Range for scaling
    activation_range = global_max_abs - global_min_abs
    if activation_range < 1e-6:
        activation_range = 1e-6  # Avoid division by zero
    
    def build_token_spans(tokens, activations, start_idx, end_idx, global_min_abs, global_max_abs, activation_range):
        """Build token spans for a range of indices."""
        spans = []
        for i in range(start_idx, end_idx):
            tok = tokens[i] if i < len(tokens) else ""
            act = activations[i] if i < len(activations) else 0
            
            # Sanitize activation value (handle NaN/Inf)
            if math.isnan(act) or math.isinf(act):
                act = 0.0
            
            # Color intensity scaled to [min_abs, max_abs] range
            # Dimmest color = min_abs, brightest = max_abs
            abs_act = abs(act)
            # Map to 0-1 range based on where it falls in [min_abs, max_abs]
            intensity = (abs_act - global_min_abs) / activation_range if activation_range > 0 else 0
            intensity = max(0.0, min(1.0, intensity))  # Clamp to [0, 1]
            
            # Apply minimum visibility (0.15) so even dimmest tokens are visible
            # Scale intensity from [0.15, 0.85] for good contrast
            visual_intensity = 0.15 + intensity * 0.70
            
            if act > 0:
                bg_color = f"rgba(100, 150, 255, {visual_intensity:.2f})"
            else:
                bg_color = f"rgba(255, 100, 100, {visual_intensity:.2f})"
            
            tok_escaped = _escape_token(tok)
            
            spans.append(
                f'<span class="token-span" data-act="{act:.4f}" '
                f'style="background:{bg_color};padding:2px 3px;border-radius:2px;margin:0 1px;display:inline;">'
                f'{tok_escaped}</span>'
            )
        return "".join(spans)
    
    # Check if this is a task dashboard (random samples) vs data dashboard (top/bottom)
    is_random_samples = len(dashboard.bottom_examples) == 0 and len(dashboard.top_examples) > 0
    
    if is_random_samples:
        # Task dashboard: show random samples
        html_parts.append(
            '<div style="margin-bottom:8px;color:#43aa8b;font-size:11px;font-weight:bold;">'
            f'Random Samples ({len(dashboard.top_examples)}):</div>'
        )
        
        for idx, ex in enumerate(dashboard.top_examples[:20]):
            tokens = ex.tokens
            activations = ex.activations
            
            # For random samples, show full context (no windowing)
            full_spans = build_token_spans(tokens, activations, 0, len(tokens), global_min_abs, global_max_abs, activation_range)
            
            # Container showing full context (no numbers or borders for task dashboards)
            html_parts.append(
                f'<div class="example-container" style="margin:4px 0;font-size:12px;line-height:1.5;font-family:monospace;">'
                f'<span class="full-view">{full_spans}</span>'
                f'</div>'
            )
    else:
        # Data dashboard: show top activating examples (positive)
        if dashboard.top_examples:
            html_parts.append(
                '<div style="margin-bottom:8px;color:#4a90d9;font-size:11px;font-weight:bold;">'
                'Top Activations (positive):</div>'
            )
            
            for idx, ex in enumerate(dashboard.top_examples[:10]):
                tokens = ex.tokens
                activations = ex.activations
                max_idx = ex.max_position
                max_act = ex.max_activation
                
                # Window around max activation
                win_start = max(0, max_idx - window_size)
                win_end = min(len(tokens), max_idx + window_size + 1)
                
                # Build windowed token spans
                windowed_spans = build_token_spans(tokens, activations, win_start, win_end, global_min_abs, global_max_abs, activation_range)
                
                # Build full context token spans
                full_spans = build_token_spans(tokens, activations, 0, len(tokens), global_min_abs, global_max_abs, activation_range)
                
                # Container with both windowed and full views
                html_parts.append(
                    f'<div class="example-container" style="margin:6px 0;font-size:12px;line-height:1.5;font-family:monospace;">'
                    f'<span style="color:#4a4;font-size:10px;margin-right:4px;">[+{max_act:.2f}]</span>'
                    f'<span class="windowed-view">{windowed_spans}</span>'
                    f'<span class="full-view" style="display:none;">{full_spans}</span>'
                    f'</div>'
                )
        
        # Bottom activating examples (negative)
        if dashboard.bottom_examples:
            html_parts.append(
                '<div style="margin-top:12px;margin-bottom:8px;color:#e94560;font-size:11px;font-weight:bold;">'
                'Bottom Activations (negative):</div>'
            )
            
            for idx, ex in enumerate(dashboard.bottom_examples[:10]):
                tokens = ex.tokens
                activations = ex.activations
                min_idx = ex.min_position
                min_act = ex.min_activation
                
                # Window around min activation
                win_start = max(0, min_idx - window_size)
                win_end = min(len(tokens), min_idx + window_size + 1)
                
                # Build windowed token spans
                windowed_spans = build_token_spans(tokens, activations, win_start, win_end, global_min_abs, global_max_abs, activation_range)
                
                # Build full context token spans
                full_spans = build_token_spans(tokens, activations, 0, len(tokens), global_min_abs, global_max_abs, activation_range)
                
                # Container with both windowed and full views
                html_parts.append(
                    f'<div class="example-container" style="margin:6px 0;font-size:12px;line-height:1.5;font-family:monospace;">'
                    f'<span style="color:#a44;font-size:10px;margin-right:4px;">[{min_act:.2f}]</span>'
                    f'<span class="windowed-view">{windowed_spans}</span>'
                    f'<span class="full-view" style="display:none;">{full_spans}</span>'
                    f'</div>'
                )
    
    if not dashboard.top_examples and not dashboard.bottom_examples:
        html_parts.append('<div class="no-data">No activation examples</div>')
    
    return "".join(html_parts)


def _escape_token(tok: str) -> str:
    """Escape a token for HTML display."""
    # Handle EOS token and other special tokens BEFORE escaping - display inline without newlines
    # The EOS token string from GPT-2 tokenizer is literally "<|endoftext|>"
    if "<|endoftext|>" in tok:
        tok = tok.replace("<|endoftext|>", "⏹")
    
    # Remove special characters that appear as hashtags in some tokenizers
    # GPT-2 uses "Ġ" (U+0120) to represent a leading space, and "Ċ" (U+010A) for newlines
    # Also handle the common "##" prefix used by some tokenizers
    tok = tok.replace("Ġ", " ")  # GPT-2 space marker
    tok = tok.replace("Ċ", "")   # GPT-2 newline marker
    tok = tok.replace("##", "")  # BERT-style subword marker
    tok = tok.replace("▁", " ")  # SentencePiece space marker
    
    # Remove ALL newline-like characters BEFORE HTML escaping to prevent line breaks
    tok = tok.replace("\n", "↵")
    tok = tok.replace("\r", "")
    tok = tok.replace("\x0b", "")  # Vertical tab
    tok = tok.replace("\x0c", "")  # Form feed
    tok = tok.replace("\x85", "")  # Next line (NEL)
    tok = tok.replace("\u2028", "")  # Line separator
    tok = tok.replace("\u2029", "")  # Paragraph separator
    
    tok_escaped = (
        tok.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
        .replace("\t", "→")
    )
    
    if tok_escaped == " " or tok_escaped == "":
        tok_escaped = "·"
    return tok_escaped


def _get_html_template() -> str:
    """Return the main HTML template for the circuit viewer."""
    return '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Circuit Viewer</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            overflow: hidden;
        }
        #container {
            display: flex;
            height: 100vh;
            width: 100vw;
        }
        #graph-container {
            flex: 1;
            position: relative;
            overflow: hidden;
            background: #1e1e2f;
        }
        #sidebar {
            width: 350px;
            background: #16213e;
            border-left: 1px solid #333;
            padding: 15px;
            overflow-y: auto;
        }
        .header {
            padding: 10px 15px;
            background: #0f3460;
            border-bottom: 1px solid #333;
            font-size: 14px;
            color: #eee;
        }
        svg {
            width: 100%;
            height: 100%;
        }
        
        /* Node styles by location type */
        .node { cursor: pointer; }
        .node circle, .node rect, .node path {
            stroke: #555;
            stroke-width: 1.5px;
        }
        .node.hovered circle, .node.hovered rect, .node.hovered path { 
            stroke: #fff; stroke-width: 3px; 
        }
        .node.selected circle, .node.selected rect, .node.selected path { 
            stroke: #ff0; stroke-width: 3px; 
        }
        .node text {
            font-size: 9px;
            fill: #888;
            pointer-events: none;
        }
        
        /* Node colors by location */
        .node-attn_in circle { fill: #6c5ce7; }
        .node-attn_q circle { fill: #e17055; }
        .node-attn_k circle { fill: #fdcb6e; }
        .node-attn_v circle { fill: #00b894; }
        .node-attn_out circle { fill: #0984e3; }
        .node-mlp_in circle { fill: #e84393; }
        .node-mlp_neuron circle { fill: #d63031; }
        .node-mlp_out circle { fill: #fd79a8; }
        .node-resid_pre rect { fill: #74b9ff; }
        .node-resid_mid rect { fill: #55efc4; }
        .node-resid_post rect { fill: #ffeaa7; }
        
        /* Edge styles */
        .link {
            fill: none;
            stroke-opacity: 0.4;
        }
        .link.highlighted {
            stroke-opacity: 1;
            stroke-width: 2px !important;
        }
        .link.positive { stroke: #4a90d9; }
        .link.negative { stroke: #e94560; }
        
        /* Layer zone dividers */
        .zone-divider {
            stroke: #333;
            stroke-width: 1;
            stroke-dasharray: 4,4;
        }
        .zone-label {
            fill: #666;
            font-size: 10px;
        }
        
        /* Dashboard tooltip */
        #dashboard-tooltip {
            position: fixed;
            background: #1e1e2f;
            border: 1px solid #444;
            border-radius: 8px;
            padding: 15px;
            width: 800px;
            max-height: 400px;
            overflow-y: auto;
            z-index: 1000;
            display: none;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            font-size: 12px;
            color: #eee;
            left: 50%;
            transform: translateX(-50%);
            bottom: 10px;
        }
        #dashboard-tooltip.visible { display: block; }
        #dashboard-tooltip.pinned { border: 2px solid #4a90d9; }
        #dashboard-tooltip .close-btn {
            position: absolute;
            top: 8px;
            right: 10px;
            cursor: pointer;
            color: #888;
            font-size: 16px;
        }
        #dashboard-tooltip .close-btn:hover { color: #fff; }
        #dashboard-tooltip .token-span:hover {
            outline: 2px solid #fff;
            cursor: pointer;
        }
        .tooltip-header {
            font-weight: bold;
            margin-bottom: 8px;
            padding-bottom: 6px;
            border-bottom: 1px solid #444;
            color: #4a90d9;
        }
        .no-data {
            color: #888;
            font-style: italic;
        }
        
        /* Sidebar */
        .section-title {
            font-size: 14px;
            font-weight: bold;
            margin: 15px 0 10px;
            color: #4a90d9;
        }
        .node-info {
            background: #222;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .node-info-row {
            display: flex;
            justify-content: space-between;
            margin: 4px 0;
            font-size: 12px;
        }
        .node-info-label { color: #888; }
        .connections-list {
            max-height: 250px;
            overflow-y: auto;
        }
        .connection-item {
            display: flex;
            justify-content: space-between;
            padding: 4px 8px;
            background: #222;
            margin: 2px 0;
            border-radius: 3px;
            font-size: 11px;
            cursor: pointer;
        }
        .connection-item:hover { background: #333; }
        .connection-weight { font-family: monospace; }
        .connection-weight.positive { color: #4a90d9; }
        .connection-weight.negative { color: #e94560; }
        
        /* Controls */
        .controls { margin-bottom: 15px; }
        .controls label {
            display: block;
            margin: 5px 0;
            font-size: 12px;
        }
        .controls input[type="range"] { width: 100%; }
        
        /* Legend */
        .legend {
            margin-top: 20px;
            padding: 10px;
            background: #222;
            border-radius: 5px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin: 4px 0;
            font-size: 11px;
        }
        .legend-color {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        /* Zoom controls */
        .zoom-controls {
            position: absolute;
            bottom: 20px;
            right: 20px;
            display: flex;
            gap: 5px;
        }
        .zoom-btn {
            width: 30px;
            height: 30px;
            background: #333;
            border: 1px solid #555;
            color: #fff;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        .zoom-btn:hover { background: #444; }
        
        /* Summary stats */
        .summary-stats {
            background: #222;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div id="container">
        <div id="graph-container">
            <div class="header">
                <strong>Circuit Graph</strong> - <span id="graph-summary"></span>
            </div>
            <svg id="graph-svg"></svg>
            <div class="zoom-controls">
                <button class="zoom-btn" onclick="zoomIn()">+</button>
                <button class="zoom-btn" onclick="zoomOut()">−</button>
                <button class="zoom-btn" onclick="resetZoom()">⟲</button>
            </div>
        </div>
        <div id="sidebar">
            <div class="summary-stats" id="summary-stats"></div>
            <div class="controls">
                <label>
                    Link opacity: <span id="opacity-val">0.4</span>
                    <input type="range" id="link-opacity" min="0.1" max="1" step="0.1" value="0.4">
                </label>
                <label>
                    Min edge weight: <span id="edge-thresh-val">0.0</span>
                    <input type="range" id="edge-threshold" min="0" max="1" step="0.01" value="0">
                </label>
            </div>
            <div id="selected-node-info"></div>
            <div class="section-title">Incoming Connections</div>
            <div class="connections-list" id="incoming-connections"></div>
            <div class="section-title">Outgoing Connections</div>
            <div class="connections-list" id="outgoing-connections"></div>
            
            <div class="legend">
                <div style="font-weight:bold;margin-bottom:8px;">Node Types</div>
                <div class="legend-item"><div class="legend-color" style="background:#6c5ce7;"></div>attn_in</div>
                <div class="legend-item"><div class="legend-color" style="background:#e17055;"></div>attn_q</div>
                <div class="legend-item"><div class="legend-color" style="background:#fdcb6e;"></div>attn_k</div>
                <div class="legend-item"><div class="legend-color" style="background:#00b894;"></div>attn_v</div>
                <div class="legend-item"><div class="legend-color" style="background:#0984e3;"></div>attn_out</div>
                <div class="legend-item"><div class="legend-color" style="background:#e84393;"></div>mlp_in</div>
                <div class="legend-item"><div class="legend-color" style="background:#d63031;"></div>mlp_neuron</div>
                <div class="legend-item"><div class="legend-color" style="background:#fd79a8;"></div>mlp_out</div>
                <div class="legend-item"><div class="legend-color" style="background:#74b9ff;border-radius:2px;"></div>resid_pre</div>
                <div class="legend-item"><div class="legend-color" style="background:#55efc4;border-radius:2px;"></div>resid_mid</div>
                <div class="legend-item"><div class="legend-color" style="background:#ffeaa7;border-radius:2px;"></div>resid_post</div>
            </div>
        </div>
    </div>
    <div id="dashboard-tooltip"></div>
    
    <script>
        // Data will be injected here
        const graphData = __GRAPH_DATA__;
        const dashboardData = __DASHBOARD_DATA__;
        
        // Layout configuration
        const LOCATION_ORDER = [
            'resid_pre', 'attn_in', 'attn_q', 'attn_k', 'attn_v', 'attn_out',
            'resid_mid', 'mlp_in', 'mlp_neuron', 'mlp_out', 'resid_post'
        ];
        const ZONE_HEIGHT = 40;  // Height per location zone
        const LAYER_GAP = 30;  // Gap between layers
        const NODE_SPACING = 15;  // Horizontal spacing between nodes
        const MARGIN = { top: 50, right: 50, bottom: 50, left: 80 };
        
        // Initialize
        const svg = d3.select("#graph-svg");
        const container = d3.select("#graph-container");
        const tooltip = d3.select("#dashboard-tooltip");
        
        let width = container.node().clientWidth;
        let height = container.node().clientHeight - 40;
        
        const g = svg.append("g");
        
        // Zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => {
                g.attr("transform", event.transform);
            });
        svg.call(zoom);
        
        function zoomIn() { svg.transition().call(zoom.scaleBy, 1.3); }
        function zoomOut() { svg.transition().call(zoom.scaleBy, 0.7); }
        function resetZoom() { svg.transition().call(zoom.transform, d3.zoomIdentity); }
        
        // Parse nodes and edges
        const nodes = graphData.nodes;
        const edges = graphData.edges;
        const metadata = graphData.metadata;
        
        // Update summary
        document.getElementById("graph-summary").textContent = 
            `${nodes.length} nodes, ${edges.length} edges, ${metadata.n_layers} layers`;
        
        document.getElementById("summary-stats").innerHTML = `
            <div><strong>Nodes:</strong> ${nodes.length}</div>
            <div><strong>Edges:</strong> ${edges.length}</div>
            <div><strong>Layers:</strong> ${metadata.n_layers}</div>
            <div><strong>d_model:</strong> ${metadata.d_model}</div>
            <div><strong>d_mlp:</strong> ${metadata.d_mlp}</div>
        `;
        
        // Create node lookup
        const nodeById = new Map(nodes.map(n => [n.node_id, n]));
        
        // Add incoming/outgoing edges to nodes
        nodes.forEach(n => {
            n.incoming = edges.filter(e => e.target_id === n.node_id);
            n.outgoing = edges.filter(e => e.source_id === n.node_id);
        });
        
        // Calculate positions
        // Y: layer (bottom to top), then location zone within layer (bottom to top)
        // X: spread nodes horizontally within each zone
        
        const layerHeight = LOCATION_ORDER.length * ZONE_HEIGHT + LAYER_GAP;
        const totalHeight = metadata.n_layers * layerHeight + MARGIN.top + MARGIN.bottom;
        
        // Group nodes by layer and location
        const nodeGroups = {};
        nodes.forEach(n => {
            const key = `${n.layer}_${n.location}`;
            if (!nodeGroups[key]) nodeGroups[key] = [];
            nodeGroups[key].push(n);
        });
        
        // Calculate max width needed
        let maxNodesInZone = 0;
        Object.values(nodeGroups).forEach(group => {
            maxNodesInZone = Math.max(maxNodesInZone, group.length);
        });
        const totalWidth = Math.max(maxNodesInZone * NODE_SPACING + MARGIN.left + MARGIN.right, width);
        
        // Position nodes
        nodes.forEach(n => {
            const locIndex = LOCATION_ORDER.indexOf(n.location);
            if (locIndex === -1) return;
            
            // Y position: layer * layerHeight + location zone offset
            // Bottom layer (0) is at the bottom of the SVG
            const layerY = totalHeight - MARGIN.bottom - (n.layer + 1) * layerHeight;
            const zoneY = layerY + (LOCATION_ORDER.length - 1 - locIndex) * ZONE_HEIGHT;
            n.y = zoneY + ZONE_HEIGHT / 2;
            
            // X position: spread within zone
            const key = `${n.layer}_${n.location}`;
            const group = nodeGroups[key];
            const idx = group.indexOf(n);
            const groupWidth = group.length * NODE_SPACING;
            const startX = MARGIN.left + (totalWidth - MARGIN.left - MARGIN.right - groupWidth) / 2;
            n.x = startX + idx * NODE_SPACING + NODE_SPACING / 2;
        });
        
        // Draw layer backgrounds and zone dividers
        const layerBg = g.append("g").attr("class", "layer-backgrounds");
        
        for (let layer = 0; layer < metadata.n_layers; layer++) {
            const layerY = totalHeight - MARGIN.bottom - (layer + 1) * layerHeight;
            
            // Layer label
            layerBg.append("text")
                .attr("x", 15)
                .attr("y", layerY + layerHeight / 2)
                .attr("fill", "#666")
                .attr("font-size", "12px")
                .attr("dominant-baseline", "middle")
                .text(`L${layer}`);
            
            // Zone dividers (dotted lines between zones)
            for (let z = 0; z < LOCATION_ORDER.length; z++) {
                const zoneY = layerY + z * ZONE_HEIGHT;
                
                layerBg.append("line")
                    .attr("class", "zone-divider")
                    .attr("x1", MARGIN.left - 10)
                    .attr("x2", totalWidth - MARGIN.right + 10)
                    .attr("y1", zoneY)
                    .attr("y2", zoneY);
            }
            
            // Layer bottom divider (solid)
            layerBg.append("line")
                .attr("x1", MARGIN.left - 10)
                .attr("x2", totalWidth - MARGIN.right + 10)
                .attr("y1", layerY + layerHeight - LAYER_GAP / 2)
                .attr("y2", layerY + layerHeight - LAYER_GAP / 2)
                .attr("stroke", "#444")
                .attr("stroke-width", 2);
        }
        
        // Draw edges
        const linkElements = g.append("g")
            .attr("class", "links")
            .selectAll("path")
            .data(edges)
            .join("path")
            .attr("class", d => `link ${d.weight >= 0 ? 'positive' : 'negative'}`)
            .attr("stroke-width", d => Math.max(0.3, Math.min(3, Math.abs(d.weight) * 2)))
            .attr("d", d => {
                const src = nodeById.get(d.source_id);
                const tgt = nodeById.get(d.target_id);
                if (!src || !tgt) return "";
                return `M${src.x},${src.y}L${tgt.x},${tgt.y}`;
            });
        
        // Draw nodes
        const nodeElements = g.append("g")
            .attr("class", "nodes")
            .selectAll("g")
            .data(nodes)
            .join("g")
            .attr("class", d => `node node-${d.location}`)
            .attr("transform", d => `translate(${d.x},${d.y})`)
            .on("click", (event, d) => {
                selectNode(d);
                showDashboardTooltip(d, event, true);
            })
            .on("mouseenter", (event, d) => hoverNode(event, d))
            .on("mouseleave", () => unhoverNode());
        
        // Different shapes for different node types
        nodeElements.each(function(d) {
            const el = d3.select(this);
            if (d.location.startsWith("resid")) {
                // Squares for residual nodes
                el.append("rect")
                    .attr("x", -4)
                    .attr("y", -4)
                    .attr("width", 8)
                    .attr("height", 8);
            } else {
                // Circles for other nodes
                el.append("circle")
                    .attr("r", 4);
            }
        });
        
        // Interaction handlers
        let selectedNode = null;
        let hoveredNode = null;
        let tooltipPinned = false;
        
        function selectNode(node) {
            selectedNode = node;
            updateSelection();
            showNodeInfo(node);
        }
        
        function hoverNode(event, node) {
            hoveredNode = node;
            nodeElements.classed("hovered", d => d === node);
            
            // Highlight connected links
            linkElements.classed("highlighted", l => 
                l.source_id === node.node_id || l.target_id === node.node_id
            );
            
            // Show dashboard tooltip (hover only, not pinned)
            if (!tooltipPinned) {
                showDashboardTooltip(node, event, false);
            }
        }
        
        function showDashboardTooltip(node, event, pinned) {
            const dashData = dashboardData[node.node_id];
            
            let html = pinned ? '<span class="close-btn" onclick="closeDashboard()">✕</span>' : '';
            html += `<div class="tooltip-header">${node.location} @ Layer ${node.layer}, Index ${node.index}</div>`;
            
            if (node.head_idx !== null && node.head_idx !== undefined) {
                html += `<div style="margin-bottom:8px;color:#888;">Head ${node.head_idx}, Dim ${node.head_dim_idx}</div>`;
            }
            
            if (dashData) {
                html += dashData;
            } else {
                html += '<div class="no-data">No dashboard data available</div>';
            }
            
            if (pinned) {
                html += '<div style="margin-top:10px;color:#666;font-size:10px;">Click ✕ or press Escape to close</div>';
            }
            
            tooltip.html(html);
            tooltip.classed("visible", true)
                   .classed("pinned", pinned);
            
            tooltipPinned = pinned;
        }
        
        window.closeDashboard = function() {
            tooltip.classed("visible", false).classed("pinned", false);
            tooltipPinned = false;
        };
        
        // Close dashboard on Escape key
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape" && tooltipPinned) {
                closeDashboard();
            }
        });
        
        // Token activation tooltip
        const tokenTooltip = d3.select("body")
            .append("div")
            .attr("id", "token-act-tooltip")
            .style("position", "fixed")
            .style("background", "#2a2a40")
            .style("border", "1px solid #666")
            .style("border-radius", "4px")
            .style("padding", "4px 8px")
            .style("font-size", "12px")
            .style("font-family", "monospace")
            .style("color", "#fff")
            .style("pointer-events", "none")
            .style("z-index", "1001")
            .style("display", "none");
        
        document.addEventListener("mouseover", (e) => {
            if (e.target.classList.contains("token-span")) {
                const act = e.target.dataset.act;
                if (act !== undefined) {
                    tokenTooltip
                        .style("display", "block")
                        .style("left", (e.pageX + 10) + "px")
                        .style("top", (e.pageY - 25) + "px")
                        .html(`activation: <b>${parseFloat(act).toFixed(4)}</b>`);
                }
            }
        });
        
        document.addEventListener("mouseout", (e) => {
            if (e.target.classList.contains("token-span")) {
                tokenTooltip.style("display", "none");
            }
        });
        
        document.addEventListener("mousemove", (e) => {
            if (e.target.classList.contains("token-span")) {
                tokenTooltip
                    .style("left", (e.pageX + 10) + "px")
                    .style("top", (e.pageY - 25) + "px");
            }
        });
        
        function unhoverNode() {
            hoveredNode = null;
            nodeElements.classed("hovered", false);
            if (!selectedNode) {
                linkElements.classed("highlighted", false);
            }
            if (!tooltipPinned) {
                tooltip.classed("visible", false);
            }
        }
        
        function updateSelection() {
            nodeElements.classed("selected", d => d === selectedNode);
            linkElements.classed("highlighted", l => 
                selectedNode && (l.source_id === selectedNode.node_id || l.target_id === selectedNode.node_id)
            );
        }
        
        function showNodeInfo(node) {
            const info = document.getElementById("selected-node-info");
            info.innerHTML = `
                <div class="node-info">
                    <div class="node-info-row">
                        <span class="node-info-label">Location:</span>
                        <span>${node.location}</span>
                    </div>
                    <div class="node-info-row">
                        <span class="node-info-label">Layer:</span>
                        <span>${node.layer}</span>
                    </div>
                    <div class="node-info-row">
                        <span class="node-info-label">Index:</span>
                        <span>${node.index}</span>
                    </div>
                    ${node.head_idx !== null && node.head_idx !== undefined ? `
                    <div class="node-info-row">
                        <span class="node-info-label">Head:</span>
                        <span>${node.head_idx}</span>
                    </div>
                    <div class="node-info-row">
                        <span class="node-info-label">Head Dim:</span>
                        <span>${node.head_dim_idx}</span>
                    </div>` : ''}
                </div>
            `;
            
            // Show connections
            const incoming = document.getElementById("incoming-connections");
            incoming.innerHTML = node.incoming
                .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
                .slice(0, 30)
                .map(l => {
                    const src = nodeById.get(l.source_id);
                    if (!src) return '';
                    return `<div class="connection-item" onclick="selectNodeById('${l.source_id}')">
                        <span>${src.location}[${src.index}]</span>
                        <span class="connection-weight ${l.weight >= 0 ? 'positive' : 'negative'}">${l.weight.toFixed(3)}</span>
                    </div>`;
                }).join("");
            
            const outgoing = document.getElementById("outgoing-connections");
            outgoing.innerHTML = node.outgoing
                .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
                .slice(0, 30)
                .map(l => {
                    const tgt = nodeById.get(l.target_id);
                    if (!tgt) return '';
                    return `<div class="connection-item" onclick="selectNodeById('${l.target_id}')">
                        <span>${tgt.location}[${tgt.index}]</span>
                        <span class="connection-weight ${l.weight >= 0 ? 'positive' : 'negative'}">${l.weight.toFixed(3)}</span>
                    </div>`;
                }).join("");
        }
        
        window.selectNodeById = function(nodeId) {
            const node = nodeById.get(nodeId);
            if (node) selectNode(node);
        };
        
        // Link opacity control
        document.getElementById("link-opacity").addEventListener("input", (e) => {
            const val = e.target.value;
            document.getElementById("opacity-val").textContent = val;
            linkElements.style("stroke-opacity", val);
        });
        
        // Edge threshold control
        document.getElementById("edge-threshold").addEventListener("input", (e) => {
            const thresh = parseFloat(e.target.value);
            document.getElementById("edge-thresh-val").textContent = thresh.toFixed(2);
            
            linkElements.style("display", d => 
                Math.abs(d.weight) >= thresh ? "block" : "none"
            );
        });
        
        // Initial zoom to fit
        const scale = Math.min(
            (width - 100) / totalWidth,
            (height - 100) / totalHeight,
            1.5
        );
        const offsetX = (width - totalWidth * scale) / 2;
        const offsetY = (height - totalHeight * scale) / 2;
        svg.call(zoom.transform, d3.zoomIdentity.translate(offsetX, offsetY).scale(scale));
        
        // Handle window resize
        window.addEventListener("resize", () => {
            width = container.node().clientWidth;
            height = container.node().clientHeight - 40;
        });
    </script>
</body>
</html>'''


def export_circuit_to_html(
    graph: CircuitGraph,
    output_path: str,
    dashboards: Optional[Dict[str, NodeDashboardData]] = None,
    verbose: bool = True,
):
    """
    Export a circuit graph to a self-contained HTML file.
    
    Args:
        graph: The circuit graph to visualize
        output_path: Path for the output HTML file
        dashboards: Optional pre-computed dashboard data
        verbose: Show progress
    """
    if verbose:
        print(f"Preparing graph data ({len(graph.nodes)} nodes, {len(graph.edges)} edges)...")
    
    # Convert graph to JSON format
    graph_data = graph.to_dict()
    
    # Convert dashboards to HTML
    dashboard_html = {}
    if dashboards:
        if verbose:
            print(f"Generating dashboard HTML for {len(dashboards)} nodes...")
        for node_id, dash in dashboards.items():
            dashboard_html[node_id] = _generate_dashboard_html(dash)
    
    # Generate HTML
    if verbose:
        print("Generating HTML...")
    
    html = _get_html_template()
    html = html.replace("__GRAPH_DATA__", json.dumps(graph_data))
    html = html.replace("__DASHBOARD_DATA__", json.dumps(dashboard_html))
    
    # Write file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    if verbose:
        print(f"Circuit exported to: {output_path}")
    
    return output_path


def visualize_circuit(
    checkpoint_path: str,
    output_path: str,
    model_path: Optional[str] = None,
    tokenizer_name: Optional[str] = None,
    compute_dashboards_flag: bool = True,
    data_source: str = "simplestories",
    n_samples: int = 500,
    edge_threshold: float = 0.0,
    device: str = "cuda",
    verbose: bool = True,
):
    """
    Complete pipeline to visualize a pruned circuit.
    
    Args:
        checkpoint_path: Path to pruning output directory
        output_path: Path for output HTML file
        model_path: Override model path (uses checkpoint's model_path if None)
        tokenizer_name: Override tokenizer name
        compute_dashboards_flag: Whether to compute activation dashboards
        data_source: "simplestories" or "task" for dashboard computation
        n_samples: Number of samples for dashboard computation
        edge_threshold: Minimum edge weight to include
        device: Device for computation
        verbose: Show progress
    """
    from .circuit_extract import extract_circuit
    from .dashboard import compute_dashboards as compute_dash
    from ..pruning.run_pruning import load_model
    from transformers import AutoTokenizer
    
    checkpoint_path = Path(checkpoint_path)
    
    # Extract circuit graph
    if verbose:
        print("Extracting circuit graph...")
    graph = extract_circuit(str(checkpoint_path), edge_threshold=edge_threshold, device=device)
    
    if verbose:
        summary = graph.summary()
        print(f"  Nodes: {summary['num_nodes']}")
        print(f"  Edges: {summary['num_edges']}")
    
    # Compute dashboards if requested
    dashboards = None
    if compute_dashboards_flag:
        if verbose:
            print("Loading model for dashboard computation...")
        
        # Load results to get model path
        results_path = checkpoint_path / "results.json"
        with open(results_path, "r") as f:
            results = json.load(f)
        
        if model_path is None:
            model_path = results.get("model_path")
        
        model, config_dict = load_model(model_path, device=device)
        
        # Get tokenizer
        if tokenizer_name is None:
            tokenizer_name = config_dict.get("training_config", {}).get("tokenizer_name")
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if verbose:
            print("Computing node dashboards...")
        
        dashboards = compute_dash(
            graph=graph,
            model=model,
            tokenizer=tokenizer,
            data_source=data_source,
            n_samples=n_samples,
            device=device,
            verbose=verbose,
        )
    
    # Export to HTML
    export_circuit_to_html(
        graph=graph,
        output_path=output_path,
        dashboards=dashboards,
        verbose=verbose,
    )
    
    return graph, dashboards

