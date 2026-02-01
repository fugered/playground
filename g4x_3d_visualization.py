#!/usr/bin/env python3
"""
G4X 3D Visualization Module
===========================
Interactive 3D visualization for G4X serial section reconstructions.

Provides:
- 3D scatter plots of transcripts and cells
- Section-by-section exploration
- Gene/protein expression overlays
- Niche visualization
- Export to HTML for sharing
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union
from pathlib import Path


def plot_transcripts_3d(
    transcripts: pd.DataFrame,
    color_by: str = 'gene_name',
    genes: Optional[List[str]] = None,
    point_size: float = 2.0,
    opacity: float = 0.7,
    max_points: int = 100000,
    title: str = "3D Transcript Distribution"
) -> "plotly.graph_objects.Figure":
    """
    Create interactive 3D scatter plot of transcripts.
    
    Parameters
    ----------
    transcripts : pd.DataFrame
        Transcript table with x, y, z, gene_name columns
    color_by : str
        Column to color by ('gene_name', 'section_id', 'confidence_score')
    genes : List[str], optional
        Specific genes to display (None = all)
    point_size : float
        Marker size
    opacity : float
        Marker opacity
    max_points : int
        Maximum points to display (subsamples if exceeded)
    title : str
        Plot title
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    import plotly.express as px
    
    df = transcripts.copy()
    
    # Filter to specific genes if requested
    if genes is not None:
        df = df[df['gene_name'].isin(genes)]
    
    # Subsample if too many points
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)
    
    # Create figure
    fig = px.scatter_3d(
        df,
        x='x',
        y='y',
        z='z',
        color=color_by,
        hover_data=['gene_name', 'section_id'] if 'section_id' in df.columns else ['gene_name'],
        title=title,
        opacity=opacity
    )
    
    fig.update_traces(marker=dict(size=point_size))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X (pixels)',
            yaxis_title='Y (pixels)',
            zaxis_title='Z (µm)',
            aspectmode='data'
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    return fig


def plot_cells_3d(
    cells: pd.DataFrame,
    color_by: str = 'niche_3d',
    size_by: Optional[str] = 'total_counts',
    point_size: float = 5.0,
    opacity: float = 0.8,
    title: str = "3D Cell Distribution"
) -> "plotly.graph_objects.Figure":
    """
    Create interactive 3D scatter plot of cells.
    
    Parameters
    ----------
    cells : pd.DataFrame
        Cell table with cell_x, cell_y, cell_z columns
    color_by : str
        Column to color by ('niche_3d', 'leiden_*', 'total_counts')
    size_by : str, optional
        Column to size points by
    point_size : float
        Base marker size
    opacity : float
        Marker opacity
    title : str
        Plot title
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    import plotly.express as px
    
    df = cells.copy()
    
    # Determine size column
    size_col = None
    if size_by and size_by in df.columns:
        # Normalize sizes
        sizes = df[size_by].values
        sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-6)
        sizes = sizes * point_size * 2 + point_size
        size_col = sizes
    
    # Handle color column
    if color_by not in df.columns:
        # Find a suitable cluster column
        for col in df.columns:
            if 'leiden' in col.lower() or 'cluster' in col.lower() or 'niche' in col.lower():
                color_by = col
                break
        else:
            color_by = 'section_index'
    
    # Create figure
    fig = px.scatter_3d(
        df,
        x='cell_x',
        y='cell_y',
        z='cell_z',
        color=color_by,
        size=size_col if size_col is not None else None,
        hover_data=['total_counts', 'n_genes_by_counts'] if 'total_counts' in df.columns else None,
        title=title,
        opacity=opacity
    )
    
    if size_col is None:
        fig.update_traces(marker=dict(size=point_size))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X (pixels)',
            yaxis_title='Y (pixels)',
            zaxis_title='Z (µm)',
            aspectmode='data'
        )
    )
    
    return fig


def plot_section_comparison(
    cells: pd.DataFrame,
    feature: str = 'total_counts',
    n_cols: int = 3
) -> "plotly.graph_objects.Figure":
    """
    Create grid of 2D section views for comparison.
    
    Parameters
    ----------
    cells : pd.DataFrame
        Cell table with section information
    feature : str
        Feature to visualize
    n_cols : int
        Number of columns in grid
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    sections = cells['section_index'].unique()
    sections = sorted(sections)
    n_sections = len(sections)
    n_rows = (n_sections + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"Section {i}" for i in sections],
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )
    
    for idx, section_idx in enumerate(sections):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        section_cells = cells[cells['section_index'] == section_idx]
        
        fig.add_trace(
            go.Scatter(
                x=section_cells['cell_x'],
                y=section_cells['cell_y'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=section_cells[feature] if feature in section_cells.columns else 'blue',
                    colorscale='Viridis',
                    showscale=(idx == 0)
                ),
                name=f"Section {section_idx}"
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title=f"Section Comparison: {feature}",
        showlegend=False,
        height=300 * n_rows
    )
    
    return fig


def plot_gene_expression_3d(
    transcripts: pd.DataFrame,
    cells: pd.DataFrame,
    gene: str,
    aggregation: str = 'count'
) -> "plotly.graph_objects.Figure":
    """
    Plot 3D gene expression at cell level.
    
    Parameters
    ----------
    transcripts : pd.DataFrame
        Transcript table
    cells : pd.DataFrame
        Cell table
    gene : str
        Gene name to visualize
    aggregation : str
        'count' or 'density'
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.express as px
    
    # Filter transcripts for gene
    gene_tx = transcripts[transcripts['gene_name'] == gene]
    
    if len(gene_tx) == 0:
        raise ValueError(f"Gene '{gene}' not found in transcripts")
    
    # Aggregate to cell level
    if 'cell_id_global' in gene_tx.columns and 'cell_id_global' in cells.columns:
        gene_counts = gene_tx.groupby('cell_id_global').size()
        
        df = cells.copy()
        df['gene_count'] = df['cell_id_global'].map(gene_counts).fillna(0)
        
        if aggregation == 'density':
            df['gene_expr'] = df['gene_count'] / df['nuclei_expanded_area']
        else:
            df['gene_expr'] = df['gene_count']
        
        fig = px.scatter_3d(
            df,
            x='cell_x',
            y='cell_y',
            z='cell_z',
            color='gene_expr',
            color_continuous_scale='Reds',
            title=f"{gene} Expression (3D)",
            opacity=0.7
        )
    else:
        # Plot transcript locations directly
        fig = px.scatter_3d(
            gene_tx,
            x='x',
            y='y',
            z='z',
            title=f"{gene} Transcripts (3D)",
            opacity=0.7
        )
    
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        scene=dict(aspectmode='data')
    )
    
    return fig


def plot_niche_distribution(
    cells: pd.DataFrame,
    niche_col: str = 'niche_3d'
) -> "plotly.graph_objects.Figure":
    """
    Plot 3D niche distribution with statistics.
    
    Parameters
    ----------
    cells : pd.DataFrame
        Cell table with niche assignments
    niche_col : str
        Column containing niche labels
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    if niche_col not in cells.columns:
        raise ValueError(f"Column '{niche_col}' not found in cells")
    
    # Create subplot with 3D view and bar chart
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "bar"}]],
        column_widths=[0.7, 0.3],
        subplot_titles=["3D Niche Distribution", "Niche Sizes"]
    )
    
    # 3D scatter
    niches = cells[niche_col].unique()
    colors = px.colors.qualitative.Set1
    
    for i, niche in enumerate(sorted(niches)):
        niche_cells = cells[cells[niche_col] == niche]
        fig.add_trace(
            go.Scatter3d(
                x=niche_cells['cell_x'],
                y=niche_cells['cell_y'],
                z=niche_cells['cell_z'],
                mode='markers',
                marker=dict(size=4, color=colors[i % len(colors)]),
                name=f"Niche {niche}",
                legendgroup=f"niche_{niche}"
            ),
            row=1, col=1
        )
    
    # Bar chart of niche sizes
    niche_counts = cells[niche_col].value_counts().sort_index()
    fig.add_trace(
        go.Bar(
            x=[f"Niche {n}" for n in niche_counts.index],
            y=niche_counts.values,
            marker_color=[colors[i % len(colors)] for i in range(len(niche_counts))],
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="3D Niche Analysis",
        height=600
    )
    
    return fig


def plot_alignment_quality(
    sections: List,
    metric: str = 'centroid_shift'
) -> "plotly.graph_objects.Figure":
    """
    Visualize alignment quality across sections.
    
    Parameters
    ----------
    sections : List[SectionData]
        List of section objects with transforms
    metric : str
        Quality metric to display
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Alignment Shifts", "Section Overlap"]
    )
    
    # Extract shifts
    shifts_x = []
    shifts_y = []
    section_ids = []
    
    for section in sections:
        if section.transform_matrix is not None:
            shifts_x.append(section.transform_matrix[0, 2])
            shifts_y.append(section.transform_matrix[1, 2])
            section_ids.append(section.section_index)
    
    # Plot shifts
    fig.add_trace(
        go.Scatter(
            x=section_ids,
            y=shifts_x,
            mode='lines+markers',
            name='X shift',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=section_ids,
            y=shifts_y,
            mode='lines+markers',
            name='Y shift',
            line=dict(color='red')
        ),
        row=1, col=1
    )
    
    # Plot section centroids to show alignment
    for section in sections:
        if section.transcripts is not None:
            cx = section.transcripts['x'].mean()
            cy = section.transcripts['y'].mean()
            
            if section.transform_matrix is not None:
                cx += section.transform_matrix[0, 2]
                cy += section.transform_matrix[1, 2]
            
            fig.add_trace(
                go.Scatter(
                    x=[cx],
                    y=[cy],
                    mode='markers+text',
                    marker=dict(size=15),
                    text=[f"S{section.section_index}"],
                    textposition="top center",
                    name=f"Section {section.section_index}",
                    showlegend=False
                ),
                row=1, col=2
            )
    
    fig.update_layout(
        title="Alignment Quality",
        height=400
    )
    
    fig.update_xaxes(title_text="Section Index", row=1, col=1)
    fig.update_yaxes(title_text="Shift (pixels)", row=1, col=1)
    fig.update_xaxes(title_text="X (pixels)", row=1, col=2)
    fig.update_yaxes(title_text="Y (pixels)", row=1, col=2)
    
    return fig


def create_dashboard(
    reconstructor,
    output_path: str = "g4x_3d_dashboard.html"
) -> str:
    """
    Create a comprehensive HTML dashboard for the 3D reconstruction.
    
    Parameters
    ----------
    reconstructor : G4X3DReconstructor
        Fitted reconstructor object
    output_path : str
        Path to save HTML file
    
    Returns
    -------
    str
        Path to saved dashboard
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create multi-panel figure
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "scatter3d"}, {"type": "scatter3d"}],
            [{"type": "scatter"}, {"type": "bar"}]
        ],
        subplot_titles=[
            "3D Transcripts",
            "3D Cells by Niche",
            "Section Overview",
            "Statistics"
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # Panel 1: 3D Transcripts (subsampled)
    if reconstructor.transcripts_3d is not None:
        tx = reconstructor.transcripts_3d.sample(n=min(50000, len(reconstructor.transcripts_3d)))
        
        # Color by section
        sections = tx['section_index'].unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
        
        for i, sec in enumerate(sorted(sections)):
            sec_tx = tx[tx['section_index'] == sec]
            fig.add_trace(
                go.Scatter3d(
                    x=sec_tx['x'],
                    y=sec_tx['y'],
                    z=sec_tx['z'],
                    mode='markers',
                    marker=dict(size=1.5, color=colors[i % len(colors)], opacity=0.5),
                    name=f"Section {sec}",
                    legendgroup="transcripts"
                ),
                row=1, col=1
            )
    
    # Panel 2: 3D Cells by niche
    if reconstructor.cells_3d is not None:
        cells = reconstructor.cells_3d
        
        niche_col = 'niche_3d' if 'niche_3d' in cells.columns else 'section_index'
        niches = cells[niche_col].unique()
        
        for i, niche in enumerate(sorted(niches)):
            niche_cells = cells[cells[niche_col] == niche]
            fig.add_trace(
                go.Scatter3d(
                    x=niche_cells['cell_x'],
                    y=niche_cells['cell_y'],
                    z=niche_cells['cell_z'],
                    mode='markers',
                    marker=dict(size=3, opacity=0.6),
                    name=f"Niche {niche}",
                    legendgroup="cells"
                ),
                row=1, col=2
            )
    
    # Panel 3: Section overview (2D)
    if reconstructor.cells_3d is not None:
        for i, section in enumerate(reconstructor.sections):
            if section.cell_metadata is not None:
                sec_cells = section.cell_metadata
                fig.add_trace(
                    go.Scatter(
                        x=sec_cells['cell_x'] + (section.transform_matrix[0, 2] if section.transform_matrix is not None else 0),
                        y=[i] * len(sec_cells) + np.random.randn(len(sec_cells)) * 0.1,
                        mode='markers',
                        marker=dict(size=2, opacity=0.3),
                        name=f"S{i}",
                        showlegend=False
                    ),
                    row=2, col=1
                )
    
    # Panel 4: Statistics
    stats = []
    labels = []
    
    if reconstructor.transcripts_3d is not None:
        stats.append(len(reconstructor.transcripts_3d))
        labels.append("Total\nTranscripts")
    
    if reconstructor.cells_3d is not None:
        stats.append(len(reconstructor.cells_3d))
        labels.append("Total\nCells")
        
        if 'n_genes_by_counts' in reconstructor.cells_3d.columns:
            stats.append(reconstructor.cells_3d['n_genes_by_counts'].median())
            labels.append("Median\nGenes/Cell")
    
    stats.append(len(reconstructor.sections))
    labels.append("Sections")
    
    fig.add_trace(
        go.Bar(
            x=labels,
            y=stats,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(stats)],
            showlegend=False,
            text=[f"{s:,.0f}" for s in stats],
            textposition='outside'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="G4X 3D Reconstruction Dashboard",
            font=dict(size=24)
        ),
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )
    
    # Save to HTML
    fig.write_html(output_path, include_plotlyjs='cdn')
    
    return output_path


def save_figure(fig, path: str, format: str = 'html') -> str:
    """
    Save a plotly figure to file.
    
    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to save
    path : str
        Output path
    format : str
        'html', 'png', 'svg', 'pdf'
    
    Returns
    -------
    str
        Path to saved file
    """
    path = Path(path)
    
    if format == 'html':
        fig.write_html(str(path), include_plotlyjs='cdn')
    elif format in ['png', 'svg', 'pdf']:
        fig.write_image(str(path), format=format, scale=2)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return str(path)


if __name__ == "__main__":
    print("G4X 3D Visualization Module")
    print("="*40)
    print("\nAvailable functions:")
    print("  - plot_transcripts_3d()")
    print("  - plot_cells_3d()")
    print("  - plot_section_comparison()")
    print("  - plot_gene_expression_3d()")
    print("  - plot_niche_distribution()")
    print("  - plot_alignment_quality()")
    print("  - create_dashboard()")
