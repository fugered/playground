#!/usr/bin/env python3
"""
G4X 3D Reconstruction Web Interface
====================================
A web-based interface for the G4X 3D reconstruction pipeline.
Allows file import, processing, and interactive 3D visualization.
"""

import os
import sys
import json
import uuid
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Import the G4X modules
from g4x_3d_reconstruction import G4X3DReconstructor, SectionData
from scipy.spatial import cKDTree
from g4x_3d_visualization import (
    plot_transcripts_3d,
    plot_cells_3d,
    plot_section_comparison,
    plot_gene_expression_3d,
    plot_niche_distribution
)


# ============================================================
# Neighborhood Analysis Functions
# ============================================================

def analyze_cell_neighborhoods(
    cells_df: pd.DataFrame,
    cell_type_col: str = 'niche_3d',
    radius: float = 100.0,
    k_neighbors: int = 15
) -> Dict:
    """
    Analyze cell neighborhoods and compute proximity statistics.

    Returns dict with:
    - neighbors: list of neighbor indices for each cell
    - neighbor_types: composition of neighbor cell types
    - proximity_matrix: cell type co-localization matrix
    """
    coords = cells_df[['cell_x', 'cell_y', 'cell_z']].values

    # Build KD-tree for efficient neighbor search
    tree = cKDTree(coords)

    # Find neighbors within radius
    neighbors_within_radius = tree.query_ball_tree(tree, r=radius)

    # Also get k-nearest neighbors
    distances, knn_indices = tree.query(coords, k=min(k_neighbors + 1, len(coords)))

    # Get cell types
    if cell_type_col in cells_df.columns:
        cell_types = cells_df[cell_type_col].values
        unique_types = sorted(cells_df[cell_type_col].unique())
        n_types = len(unique_types)
        type_to_idx = {t: i for i, t in enumerate(unique_types)}

        # Compute proximity matrix (co-localization scores)
        proximity_matrix = np.zeros((n_types, n_types))
        type_counts = np.zeros(n_types)

        for i, neighbors in enumerate(neighbors_within_radius):
            cell_type = cell_types[i]
            type_idx = type_to_idx[cell_type]
            type_counts[type_idx] += 1

            for j in neighbors:
                if i != j:
                    neighbor_type = cell_types[j]
                    neighbor_idx = type_to_idx[neighbor_type]
                    proximity_matrix[type_idx, neighbor_idx] += 1

        # Normalize by cell type counts
        for i in range(n_types):
            if type_counts[i] > 0:
                proximity_matrix[i, :] /= type_counts[i]

        # Compute neighbor composition for each cell
        neighbor_compositions = []
        for i, neighbors in enumerate(knn_indices):
            composition = np.zeros(n_types)
            for j in neighbors[1:]:  # Skip self
                if j < len(cell_types):
                    neighbor_type = cell_types[j]
                    composition[type_to_idx[neighbor_type]] += 1
            composition /= (len(neighbors) - 1) if len(neighbors) > 1 else 1
            neighbor_compositions.append(composition)

        return {
            'neighbors_radius': neighbors_within_radius,
            'knn_indices': knn_indices.tolist(),
            'knn_distances': distances.tolist(),
            'proximity_matrix': proximity_matrix.tolist(),
            'cell_types': unique_types,
            'type_counts': type_counts.tolist(),
            'neighbor_compositions': neighbor_compositions
        }
    else:
        return {
            'neighbors_radius': neighbors_within_radius,
            'knn_indices': knn_indices.tolist(),
            'knn_distances': distances.tolist(),
            'cell_types': [],
            'proximity_matrix': []
        }


def get_cell_connectivity_edges(
    cells_df: pd.DataFrame,
    max_distance: float = 75.0,
    max_edges: int = 5000,
    cell_type_col: str = 'niche_3d',
    filter_same_type: bool = False,
    filter_different_type: bool = False
) -> Dict:
    """
    Get edges between neighboring cells for 3D visualization.

    Returns dict with edge coordinates for Plotly line traces.
    """
    coords = cells_df[['cell_x', 'cell_y', 'cell_z']].values
    n_cells = len(coords)

    # Build KD-tree
    tree = cKDTree(coords)

    # Find all pairs within distance
    pairs = tree.query_pairs(r=max_distance, output_type='ndarray')

    if len(pairs) == 0:
        return {'edges': [], 'edge_types': []}

    # Filter by cell type if requested
    if cell_type_col in cells_df.columns and (filter_same_type or filter_different_type):
        cell_types = cells_df[cell_type_col].values
        filtered_pairs = []
        for i, j in pairs:
            same_type = cell_types[i] == cell_types[j]
            if filter_same_type and same_type:
                filtered_pairs.append((i, j))
            elif filter_different_type and not same_type:
                filtered_pairs.append((i, j))
            elif not filter_same_type and not filter_different_type:
                filtered_pairs.append((i, j))
        pairs = np.array(filtered_pairs) if filtered_pairs else np.array([]).reshape(0, 2)

    # Subsample if too many edges
    if len(pairs) > max_edges:
        indices = np.random.choice(len(pairs), max_edges, replace=False)
        pairs = pairs[indices]

    # Build edge coordinates for Plotly
    edges_x = []
    edges_y = []
    edges_z = []
    edge_colors = []

    cell_types = cells_df[cell_type_col].values if cell_type_col in cells_df.columns else None

    for i, j in pairs:
        edges_x.extend([coords[i, 0], coords[j, 0], None])
        edges_y.extend([coords[i, 1], coords[j, 1], None])
        edges_z.extend([coords[i, 2], coords[j, 2], None])

        if cell_types is not None:
            # Color based on whether same or different type
            if cell_types[i] == cell_types[j]:
                edge_colors.append('same')
            else:
                edge_colors.append('different')

    return {
        'edges_x': edges_x,
        'edges_y': edges_y,
        'edges_z': edges_z,
        'edge_colors': edge_colors,
        'n_edges': len(pairs),
        'n_same_type': edge_colors.count('same') if edge_colors else 0,
        'n_different_type': edge_colors.count('different') if edge_colors else 0
    }


def compute_spatial_enrichment(
    cells_df: pd.DataFrame,
    cell_type_col: str = 'niche_3d',
    radius: float = 100.0,
    n_permutations: int = 100
) -> Dict:
    """
    Compute spatial enrichment/depletion of cell type pairs.
    Uses permutation testing to assess significance.
    """
    coords = cells_df[['cell_x', 'cell_y', 'cell_z']].values

    if cell_type_col not in cells_df.columns:
        return {'error': 'Cell type column not found'}

    cell_types = cells_df[cell_type_col].values
    unique_types = sorted(cells_df[cell_type_col].unique())
    n_types = len(unique_types)
    type_to_idx = {t: i for i, t in enumerate(unique_types)}

    # Build KD-tree
    tree = cKDTree(coords)

    # Compute observed co-localization
    observed = np.zeros((n_types, n_types))
    neighbors = tree.query_ball_tree(tree, r=radius)

    for i, neighs in enumerate(neighbors):
        type_i = type_to_idx[cell_types[i]]
        for j in neighs:
            if i != j:
                type_j = type_to_idx[cell_types[j]]
                observed[type_i, type_j] += 1

    # Permutation test
    permuted_means = np.zeros((n_types, n_types))
    permuted_stds = np.zeros((n_types, n_types))

    permuted_results = []
    for _ in range(n_permutations):
        shuffled_types = np.random.permutation(cell_types)
        perm_matrix = np.zeros((n_types, n_types))

        for i, neighs in enumerate(neighbors):
            type_i = type_to_idx[shuffled_types[i]]
            for j in neighs:
                if i != j:
                    type_j = type_to_idx[shuffled_types[j]]
                    perm_matrix[type_i, type_j] += 1

        permuted_results.append(perm_matrix)

    permuted_stack = np.stack(permuted_results)
    permuted_means = permuted_stack.mean(axis=0)
    permuted_stds = permuted_stack.std(axis=0) + 1e-6

    # Compute z-scores (enrichment/depletion)
    z_scores = (observed - permuted_means) / permuted_stds

    # Compute p-values
    p_values = np.zeros((n_types, n_types))
    for i in range(n_types):
        for j in range(n_types):
            # Two-tailed test
            count_extreme = np.sum(np.abs(permuted_stack[:, i, j] - permuted_means[i, j]) >=
                                   np.abs(observed[i, j] - permuted_means[i, j]))
            p_values[i, j] = (count_extreme + 1) / (n_permutations + 1)

    return {
        'cell_types': unique_types,
        'observed': observed.tolist(),
        'expected': permuted_means.tolist(),
        'z_scores': z_scores.tolist(),
        'p_values': p_values.tolist(),
        'enriched': (z_scores > 2).tolist(),  # Significantly enriched
        'depleted': (z_scores < -2).tolist()  # Significantly depleted
    }

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
app.config['UPLOAD_FOLDER'] = Path(tempfile.gettempdir()) / 'g4x_uploads'
app.config['RESULTS_FOLDER'] = Path(tempfile.gettempdir()) / 'g4x_results'

# Ensure directories exist
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
app.config['RESULTS_FOLDER'].mkdir(parents=True, exist_ok=True)

# Store active sessions
sessions: Dict[str, dict] = {}


@dataclass
class ProcessingStatus:
    """Track processing status for a session."""
    session_id: str
    status: str  # 'idle', 'uploading', 'processing', 'complete', 'error'
    progress: float  # 0-100
    message: str
    n_sections: int = 0
    n_transcripts: int = 0
    n_cells: int = 0
    has_results: bool = False
    error: Optional[str] = None


def get_session(session_id: str) -> dict:
    """Get or create a session."""
    if session_id not in sessions:
        sessions[session_id] = {
            'status': ProcessingStatus(
                session_id=session_id,
                status='idle',
                progress=0,
                message='Ready to upload files'
            ),
            'reconstructor': None,
            'upload_dir': app.config['UPLOAD_FOLDER'] / session_id,
            'results_dir': app.config['RESULTS_FOLDER'] / session_id,
            'sections': []
        }
        sessions[session_id]['upload_dir'].mkdir(parents=True, exist_ok=True)
        sessions[session_id]['results_dir'].mkdir(parents=True, exist_ok=True)
    return sessions[session_id]


@app.route('/')
def index():
    """Serve the main web interface."""
    session_id = str(uuid.uuid4())
    return render_template('index.html', session_id=session_id)


@app.route('/api/session/<session_id>/status')
def get_status(session_id: str):
    """Get the current processing status."""
    session = get_session(session_id)
    return jsonify(asdict(session['status']))


@app.route('/api/session/<session_id>/upload', methods=['POST'])
def upload_files(session_id: str):
    """
    Handle file uploads.
    Accepts:
    - ZIP files containing section data
    - Individual CSV/Parquet files
    - Directory structure uploads
    """
    session = get_session(session_id)
    session['status'].status = 'uploading'
    session['status'].message = 'Receiving files...'

    try:
        uploaded_files = request.files.getlist('files')
        section_index = request.form.get('section_index', 0)
        file_type = request.form.get('file_type', 'auto')

        if not uploaded_files:
            return jsonify({'error': 'No files uploaded'}), 400

        saved_files = []
        for file in uploaded_files:
            if file.filename:
                filename = secure_filename(file.filename)

                # Determine the save path based on file type
                if file_type == 'transcript_table':
                    save_dir = session['upload_dir'] / f'section_{section_index}' / 'rna'
                elif file_type == 'cell_metadata':
                    save_dir = session['upload_dir'] / f'section_{section_index}' / 'single_cell_data'
                else:
                    save_dir = session['upload_dir'] / f'section_{section_index}'

                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / filename
                file.save(save_path)
                saved_files.append(str(save_path))

        session['status'].message = f'Uploaded {len(saved_files)} file(s)'
        session['status'].progress = 10

        return jsonify({
            'success': True,
            'files': saved_files,
            'message': f'Uploaded {len(saved_files)} file(s) successfully'
        })

    except Exception as e:
        session['status'].status = 'error'
        session['status'].error = str(e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/session/<session_id>/upload-direct', methods=['POST'])
def upload_direct_data(session_id: str):
    """
    Handle direct data upload (CSV/Parquet content).
    Useful for smaller datasets or pre-processed data.
    """
    session = get_session(session_id)

    try:
        data = request.get_json()
        data_type = data.get('type')  # 'transcripts' or 'cells'
        section_index = data.get('section_index', 0)
        content = data.get('content')  # Base64 or JSON array

        save_dir = session['upload_dir'] / f'section_{section_index}'
        save_dir.mkdir(parents=True, exist_ok=True)

        if data_type == 'transcripts':
            df = pd.DataFrame(content)
            save_path = save_dir / 'rna' / 'transcript_table.csv.gz'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False, compression='gzip')
        elif data_type == 'cells':
            df = pd.DataFrame(content)
            save_path = save_dir / 'single_cell_data' / 'cell_metadata.csv.gz'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False, compression='gzip')
        else:
            return jsonify({'error': f'Unknown data type: {data_type}'}), 400

        return jsonify({
            'success': True,
            'path': str(save_path),
            'rows': len(df)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/session/<session_id>/process', methods=['POST'])
def process_data(session_id: str):
    """
    Run the 3D reconstruction pipeline on uploaded data.
    """
    session = get_session(session_id)
    session['status'].status = 'processing'
    session['status'].progress = 0

    try:
        # Get processing parameters
        params = request.get_json() or {}
        section_thickness = float(params.get('section_thickness', 5.0))
        pixel_size = float(params.get('pixel_size', 0.5))
        alignment_method = params.get('alignment_method', 'centroid')
        n_neighbors = int(params.get('n_neighbors', 30))

        # Find section directories
        upload_dir = session['upload_dir']
        section_dirs = sorted([
            d for d in upload_dir.iterdir()
            if d.is_dir() and d.name.startswith('section_')
        ])

        if not section_dirs:
            return jsonify({'error': 'No section data found. Please upload files first.'}), 400

        session['status'].message = f'Found {len(section_dirs)} sections'
        session['status'].n_sections = len(section_dirs)
        session['status'].progress = 10

        # Initialize reconstructor
        reconstructor = G4X3DReconstructor(
            section_dirs=[str(d) for d in section_dirs],
            section_thickness=section_thickness,
            pixel_size=pixel_size
        )

        # Step 1: Load sections
        session['status'].message = 'Loading sections...'
        session['status'].progress = 20
        reconstructor.load_sections(verbose=False)

        # Step 2: Align sections
        session['status'].message = 'Aligning sections...'
        session['status'].progress = 40
        reconstructor.align_sections(method=alignment_method, verbose=False)

        # Step 3: Build 3D volume
        session['status'].message = 'Building 3D volume...'
        session['status'].progress = 60
        reconstructor.build_3d_volume(verbose=False)

        # Step 4: Find niches (if enough cells)
        if reconstructor.cells_3d is not None and len(reconstructor.cells_3d) > 100:
            session['status'].message = 'Identifying 3D niches...'
            session['status'].progress = 80
            try:
                reconstructor.find_3d_niches(n_neighbors=n_neighbors, verbose=False)
            except Exception as e:
                print(f"Niche finding failed (non-fatal): {e}")

        # Step 5: Save results
        session['status'].message = 'Saving results...'
        session['status'].progress = 90
        results_dir = session['results_dir']
        reconstructor.save_results(str(results_dir), formats=['parquet'], verbose=False)

        # Update session
        session['reconstructor'] = reconstructor
        session['status'].status = 'complete'
        session['status'].progress = 100
        session['status'].message = 'Processing complete!'
        session['status'].has_results = True

        if reconstructor.transcripts_3d is not None:
            session['status'].n_transcripts = len(reconstructor.transcripts_3d)
        if reconstructor.cells_3d is not None:
            session['status'].n_cells = len(reconstructor.cells_3d)

        return jsonify({
            'success': True,
            'n_sections': len(section_dirs),
            'n_transcripts': session['status'].n_transcripts,
            'n_cells': session['status'].n_cells
        })

    except Exception as e:
        session['status'].status = 'error'
        session['status'].error = str(e)
        session['status'].message = f'Error: {str(e)}'
        traceback.print_exc()
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/session/<session_id>/visualization/<viz_type>')
def get_visualization(session_id: str, viz_type: str):
    """
    Get visualization data for Plotly.
    viz_type: 'transcripts_3d', 'cells_3d', 'sections', 'niches', 'gene'
    """
    session = get_session(session_id)

    if session['reconstructor'] is None:
        return jsonify({'error': 'No data processed yet'}), 400

    reconstructor = session['reconstructor']

    try:
        if viz_type == 'transcripts_3d':
            if reconstructor.transcripts_3d is None:
                return jsonify({'error': 'No transcript data available'}), 400

            # Subsample for web performance
            df = reconstructor.transcripts_3d
            max_points = int(request.args.get('max_points', 50000))
            if len(df) > max_points:
                df = df.sample(n=max_points, random_state=42)

            fig = plot_transcripts_3d(df, color_by='section_index', max_points=max_points)
            return jsonify(json.loads(fig.to_json()))

        elif viz_type == 'cells_3d':
            if reconstructor.cells_3d is None:
                return jsonify({'error': 'No cell data available'}), 400

            color_by = request.args.get('color_by', 'niche_3d')
            fig = plot_cells_3d(reconstructor.cells_3d, color_by=color_by)
            return jsonify(json.loads(fig.to_json()))

        elif viz_type == 'sections':
            if reconstructor.cells_3d is None:
                return jsonify({'error': 'No cell data available'}), 400

            fig = plot_section_comparison(reconstructor.cells_3d)
            return jsonify(json.loads(fig.to_json()))

        elif viz_type == 'niches':
            if reconstructor.cells_3d is None or 'niche_3d' not in reconstructor.cells_3d.columns:
                return jsonify({'error': 'No niche data available'}), 400

            fig = plot_niche_distribution(reconstructor.cells_3d)
            return jsonify(json.loads(fig.to_json()))

        elif viz_type == 'gene':
            gene = request.args.get('gene')
            if not gene:
                return jsonify({'error': 'Gene parameter required'}), 400

            if reconstructor.transcripts_3d is None:
                return jsonify({'error': 'No transcript data available'}), 400

            try:
                fig = plot_gene_expression_3d(
                    reconstructor.transcripts_3d,
                    reconstructor.cells_3d,
                    gene=gene
                )
                return jsonify(json.loads(fig.to_json()))
            except ValueError as e:
                return jsonify({'error': str(e)}), 400

        else:
            return jsonify({'error': f'Unknown visualization type: {viz_type}'}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/session/<session_id>/data/<data_type>')
def get_data(session_id: str, data_type: str):
    """
    Get raw data in JSON format for custom visualizations.
    data_type: 'transcripts', 'cells', 'genes', 'sections', 'stats'
    """
    session = get_session(session_id)

    if session['reconstructor'] is None:
        return jsonify({'error': 'No data processed yet'}), 400

    reconstructor = session['reconstructor']

    try:
        if data_type == 'transcripts':
            if reconstructor.transcripts_3d is None:
                return jsonify({'error': 'No transcript data'}), 400

            # Return subsampled data
            max_rows = int(request.args.get('max_rows', 10000))
            df = reconstructor.transcripts_3d
            if len(df) > max_rows:
                df = df.sample(n=max_rows, random_state=42)

            return jsonify({
                'columns': df.columns.tolist(),
                'data': df.to_dict(orient='records'),
                'total_rows': len(reconstructor.transcripts_3d)
            })

        elif data_type == 'cells':
            if reconstructor.cells_3d is None:
                return jsonify({'error': 'No cell data'}), 400

            max_rows = int(request.args.get('max_rows', 10000))
            df = reconstructor.cells_3d
            if len(df) > max_rows:
                df = df.sample(n=max_rows, random_state=42)

            return jsonify({
                'columns': df.columns.tolist(),
                'data': df.to_dict(orient='records'),
                'total_rows': len(reconstructor.cells_3d)
            })

        elif data_type == 'genes':
            if reconstructor.transcripts_3d is None:
                return jsonify({'error': 'No transcript data'}), 400

            genes = reconstructor.transcripts_3d['gene_name'].value_counts()
            return jsonify({
                'genes': genes.index.tolist(),
                'counts': genes.values.tolist()
            })

        elif data_type == 'sections':
            sections_info = []
            for section in reconstructor.sections:
                info = {
                    'id': section.section_id,
                    'index': section.section_index,
                    'z_position': section.z_position,
                    'n_transcripts': len(section.transcripts) if section.transcripts is not None else 0,
                    'n_cells': len(section.cell_metadata) if section.cell_metadata is not None else 0,
                }
                if section.transform_matrix is not None:
                    info['shift_x'] = section.transform_matrix[0, 2]
                    info['shift_y'] = section.transform_matrix[1, 2]
                sections_info.append(info)

            return jsonify({'sections': sections_info})

        elif data_type == 'stats':
            stats = {
                'n_sections': len(reconstructor.sections),
                'n_transcripts': len(reconstructor.transcripts_3d) if reconstructor.transcripts_3d is not None else 0,
                'n_cells': len(reconstructor.cells_3d) if reconstructor.cells_3d is not None else 0,
                'section_thickness': reconstructor.section_thickness,
                'pixel_size': reconstructor.pixel_size,
            }

            if reconstructor.transcripts_3d is not None:
                stats['n_genes'] = reconstructor.transcripts_3d['gene_name'].nunique()
                stats['volume_x'] = float(reconstructor.transcripts_3d['x'].max())
                stats['volume_y'] = float(reconstructor.transcripts_3d['y'].max())
                stats['volume_z'] = float(reconstructor.transcripts_3d['z'].max())

            if reconstructor.cells_3d is not None and 'niche_3d' in reconstructor.cells_3d.columns:
                stats['n_niches'] = int(reconstructor.cells_3d['niche_3d'].nunique())

            return jsonify(stats)

        else:
            return jsonify({'error': f'Unknown data type: {data_type}'}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/session/<session_id>/neighborhood/analysis')
def get_neighborhood_analysis(session_id: str):
    """
    Analyze cell neighborhoods and return proximity statistics.
    """
    session = get_session(session_id)

    if session['reconstructor'] is None:
        return jsonify({'error': 'No data processed yet'}), 400

    reconstructor = session['reconstructor']

    if reconstructor.cells_3d is None:
        return jsonify({'error': 'No cell data available'}), 400

    try:
        radius = float(request.args.get('radius', 100.0))
        k_neighbors = int(request.args.get('k_neighbors', 15))
        cell_type_col = request.args.get('cell_type_col', 'niche_3d')

        # Fall back to section_index if niche_3d not available
        if cell_type_col not in reconstructor.cells_3d.columns:
            cell_type_col = 'section_index'

        analysis = analyze_cell_neighborhoods(
            reconstructor.cells_3d,
            cell_type_col=cell_type_col,
            radius=radius,
            k_neighbors=k_neighbors
        )

        return jsonify(analysis)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/session/<session_id>/neighborhood/connectivity')
def get_neighborhood_connectivity(session_id: str):
    """
    Get cell connectivity edges for 3D visualization.
    """
    session = get_session(session_id)

    if session['reconstructor'] is None:
        return jsonify({'error': 'No data processed yet'}), 400

    reconstructor = session['reconstructor']

    if reconstructor.cells_3d is None:
        return jsonify({'error': 'No cell data available'}), 400

    try:
        max_distance = float(request.args.get('max_distance', 75.0))
        max_edges = int(request.args.get('max_edges', 5000))
        cell_type_col = request.args.get('cell_type_col', 'niche_3d')
        filter_same = request.args.get('filter_same', 'false').lower() == 'true'
        filter_different = request.args.get('filter_different', 'false').lower() == 'true'

        if cell_type_col not in reconstructor.cells_3d.columns:
            cell_type_col = 'section_index'

        edges = get_cell_connectivity_edges(
            reconstructor.cells_3d,
            max_distance=max_distance,
            max_edges=max_edges,
            cell_type_col=cell_type_col,
            filter_same_type=filter_same,
            filter_different_type=filter_different
        )

        return jsonify(edges)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/session/<session_id>/neighborhood/enrichment')
def get_spatial_enrichment(session_id: str):
    """
    Compute spatial enrichment analysis between cell types.
    """
    session = get_session(session_id)

    if session['reconstructor'] is None:
        return jsonify({'error': 'No data processed yet'}), 400

    reconstructor = session['reconstructor']

    if reconstructor.cells_3d is None:
        return jsonify({'error': 'No cell data available'}), 400

    try:
        radius = float(request.args.get('radius', 100.0))
        n_permutations = int(request.args.get('n_permutations', 50))
        cell_type_col = request.args.get('cell_type_col', 'niche_3d')

        if cell_type_col not in reconstructor.cells_3d.columns:
            cell_type_col = 'section_index'

        enrichment = compute_spatial_enrichment(
            reconstructor.cells_3d,
            cell_type_col=cell_type_col,
            radius=radius,
            n_permutations=n_permutations
        )

        return jsonify(enrichment)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/session/<session_id>/visualization/neighborhood')
def get_neighborhood_visualization(session_id: str):
    """
    Get combined neighborhood visualization with cells and connections.
    """
    session = get_session(session_id)

    if session['reconstructor'] is None:
        return jsonify({'error': 'No data processed yet'}), 400

    reconstructor = session['reconstructor']

    if reconstructor.cells_3d is None:
        return jsonify({'error': 'No cell data available'}), 400

    try:
        import plotly.graph_objects as go

        max_distance = float(request.args.get('max_distance', 75.0))
        max_edges = int(request.args.get('max_edges', 3000))
        max_cells = int(request.args.get('max_cells', 2000))
        cell_type_col = request.args.get('cell_type_col', 'niche_3d')
        show_edges = request.args.get('show_edges', 'true').lower() == 'true'
        edge_filter = request.args.get('edge_filter', 'all')  # 'all', 'same', 'different'

        if cell_type_col not in reconstructor.cells_3d.columns:
            cell_type_col = 'section_index'

        cells_df = reconstructor.cells_3d
        if len(cells_df) > max_cells:
            cells_df = cells_df.sample(n=max_cells, random_state=42)

        # Get cell types
        cell_types = cells_df[cell_type_col].values
        unique_types = sorted(cells_df[cell_type_col].unique())

        # Color palette
        colors = [
            '#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899',
            '#f43f5e', '#f97316', '#eab308', '#22c55e', '#14b8a6',
            '#06b6d4', '#3b82f6'
        ]

        traces = []

        # Add cell scatter traces by type
        for i, cell_type in enumerate(unique_types):
            mask = cell_types == cell_type
            type_cells = cells_df[mask]

            traces.append(go.Scatter3d(
                x=type_cells['cell_x'].values,
                y=type_cells['cell_y'].values,
                z=type_cells['cell_z'].values,
                mode='markers',
                marker=dict(
                    size=6,
                    color=colors[i % len(colors)],
                    opacity=0.8,
                    line=dict(width=1, color='white')
                ),
                name=f'Type {cell_type}',
                hovertemplate=f'Type {cell_type}<br>X: %{{x:.0f}}<br>Y: %{{y:.0f}}<br>Z: %{{z:.1f}}<extra></extra>'
            ))

        # Add connectivity edges
        if show_edges:
            filter_same = edge_filter == 'same'
            filter_different = edge_filter == 'different'

            edges = get_cell_connectivity_edges(
                cells_df,
                max_distance=max_distance,
                max_edges=max_edges,
                cell_type_col=cell_type_col,
                filter_same_type=filter_same,
                filter_different_type=filter_different
            )

            if edges['edges_x']:
                # Separate edges by type
                same_x, same_y, same_z = [], [], []
                diff_x, diff_y, diff_z = [], [], []

                edge_colors = edges.get('edge_colors', [])
                for idx in range(0, len(edges['edges_x']), 3):
                    if idx // 3 < len(edge_colors):
                        if edge_colors[idx // 3] == 'same':
                            same_x.extend(edges['edges_x'][idx:idx+3])
                            same_y.extend(edges['edges_y'][idx:idx+3])
                            same_z.extend(edges['edges_z'][idx:idx+3])
                        else:
                            diff_x.extend(edges['edges_x'][idx:idx+3])
                            diff_y.extend(edges['edges_y'][idx:idx+3])
                            diff_z.extend(edges['edges_z'][idx:idx+3])
                    else:
                        same_x.extend(edges['edges_x'][idx:idx+3])
                        same_y.extend(edges['edges_y'][idx:idx+3])
                        same_z.extend(edges['edges_z'][idx:idx+3])

                # Same-type edges (cyan)
                if same_x and edge_filter != 'different':
                    traces.append(go.Scatter3d(
                        x=same_x,
                        y=same_y,
                        z=same_z,
                        mode='lines',
                        line=dict(color='rgba(6, 182, 212, 0.4)', width=1),
                        name=f'Same-type ({edges["n_same_type"]})',
                        hoverinfo='skip'
                    ))

                # Different-type edges (pink)
                if diff_x and edge_filter != 'same':
                    traces.append(go.Scatter3d(
                        x=diff_x,
                        y=diff_y,
                        z=diff_z,
                        mode='lines',
                        line=dict(color='rgba(236, 72, 153, 0.5)', width=1.5),
                        name=f'Cross-type ({edges["n_different_type"]})',
                        hoverinfo='skip'
                    ))

        # Layout
        layout = go.Layout(
            title=dict(
                text='3D Cell Neighborhood Network',
                font=dict(size=16)
            ),
            scene=dict(
                xaxis=dict(title='X (pixels)'),
                yaxis=dict(title='Y (pixels)'),
                zaxis=dict(title='Z (um)'),
                aspectmode='data'
            ),
            legend=dict(
                yanchor='top',
                y=0.99,
                xanchor='left',
                x=0.01
            ),
            margin=dict(l=0, r=0, t=40, b=0)
        )

        fig = go.Figure(data=traces, layout=layout)
        return jsonify(json.loads(fig.to_json()))

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/session/<session_id>/visualization/proximity_matrix')
def get_proximity_matrix_viz(session_id: str):
    """
    Get proximity matrix heatmap visualization.
    """
    session = get_session(session_id)

    if session['reconstructor'] is None:
        return jsonify({'error': 'No data processed yet'}), 400

    reconstructor = session['reconstructor']

    if reconstructor.cells_3d is None:
        return jsonify({'error': 'No cell data available'}), 400

    try:
        import plotly.graph_objects as go

        radius = float(request.args.get('radius', 100.0))
        cell_type_col = request.args.get('cell_type_col', 'niche_3d')
        show_enrichment = request.args.get('show_enrichment', 'false').lower() == 'true'

        if cell_type_col not in reconstructor.cells_3d.columns:
            cell_type_col = 'section_index'

        if show_enrichment:
            # Use enrichment z-scores
            enrichment = compute_spatial_enrichment(
                reconstructor.cells_3d,
                cell_type_col=cell_type_col,
                radius=radius,
                n_permutations=50
            )
            matrix = np.array(enrichment['z_scores'])
            cell_types = enrichment['cell_types']
            title = 'Spatial Enrichment (Z-scores)'
            colorscale = 'RdBu_r'
            zmid = 0
        else:
            # Use raw proximity counts
            analysis = analyze_cell_neighborhoods(
                reconstructor.cells_3d,
                cell_type_col=cell_type_col,
                radius=radius
            )
            matrix = np.array(analysis['proximity_matrix'])
            cell_types = analysis['cell_types']
            title = 'Cell Type Proximity Matrix'
            colorscale = 'Viridis'
            zmid = None

        # Create heatmap
        labels = [f'Type {t}' for t in cell_types]

        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale=colorscale,
            zmid=zmid,
            text=np.round(matrix, 2),
            texttemplate='%{text}',
            textfont=dict(size=10),
            hovertemplate='%{y} -> %{x}<br>Value: %{z:.2f}<extra></extra>'
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis=dict(title='Neighbor Type', tickangle=45),
            yaxis=dict(title='Cell Type'),
            height=500,
            margin=dict(l=100, r=20, t=60, b=100)
        )

        return jsonify(json.loads(fig.to_json()))

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/session/<session_id>/export/<format>')
def export_data(session_id: str, format: str):
    """Export processed data in various formats."""
    session = get_session(session_id)

    if session['reconstructor'] is None:
        return jsonify({'error': 'No data processed yet'}), 400

    results_dir = session['results_dir']

    if format == 'parquet':
        # Return paths to parquet files
        files = list(results_dir.glob('*.parquet'))
        return jsonify({'files': [str(f) for f in files]})

    elif format == 'csv':
        # Generate CSV files if not already done
        reconstructor = session['reconstructor']
        reconstructor.save_results(str(results_dir), formats=['csv'], verbose=False)
        files = list(results_dir.glob('*.csv.gz'))
        return jsonify({'files': [str(f) for f in files]})

    else:
        return jsonify({'error': f'Unknown format: {format}'}), 400


@app.route('/api/session/<session_id>/download/<filename>')
def download_file(session_id: str, filename: str):
    """Download a result file."""
    session = get_session(session_id)
    results_dir = session['results_dir']

    return send_from_directory(results_dir, filename, as_attachment=True)


@app.route('/api/session/<session_id>/demo', methods=['POST'])
def load_demo_data(session_id: str):
    """Load demo/synthetic data for testing the interface."""
    session = get_session(session_id)

    try:
        session['status'].status = 'processing'
        session['status'].message = 'Generating demo data...'
        session['status'].progress = 10

        # Generate synthetic demo data
        np.random.seed(42)
        n_sections = 5
        n_transcripts_per_section = 5000
        n_cells_per_section = 500

        section_dirs = []

        for i in range(n_sections):
            section_dir = session['upload_dir'] / f'section_{i}'
            section_dir.mkdir(parents=True, exist_ok=True)

            # Generate synthetic transcript data
            rna_dir = section_dir / 'rna'
            rna_dir.mkdir(exist_ok=True)

            # Create a tissue-like distribution (circular with some structure)
            center_x, center_y = 5000, 5000
            radius = 2000

            angles = np.random.uniform(0, 2*np.pi, n_transcripts_per_section)
            radii = radius * np.sqrt(np.random.uniform(0, 1, n_transcripts_per_section))

            x = center_x + radii * np.cos(angles) + np.random.randn(n_transcripts_per_section) * 50
            y = center_y + radii * np.sin(angles) + np.random.randn(n_transcripts_per_section) * 50

            genes = np.random.choice(
                ['ACTB', 'GAPDH', 'CD3E', 'CD8A', 'CD4', 'MS4A1', 'CD68', 'KRT8', 'VIM', 'EPCAM',
                 'COL1A1', 'PECAM1', 'PTPRC', 'CD79A', 'HLA-DRA', 'S100A8', 'LYZ', 'TRAC', 'NKG7', 'FOXP3'],
                n_transcripts_per_section
            )

            transcripts = pd.DataFrame({
                'x_pixel_coordinate': x,
                'y_pixel_coordinate': y,
                'gene_name': genes,
                'cell_id': np.random.randint(0, n_cells_per_section, n_transcripts_per_section),
                'confidence_score': np.random.uniform(0.8, 1.0, n_transcripts_per_section)
            })
            transcripts.to_csv(rna_dir / 'transcript_table.csv.gz', index=False, compression='gzip')

            # Generate synthetic cell data
            cell_dir = section_dir / 'single_cell_data'
            cell_dir.mkdir(exist_ok=True)

            cell_angles = np.random.uniform(0, 2*np.pi, n_cells_per_section)
            cell_radii = radius * np.sqrt(np.random.uniform(0, 1, n_cells_per_section))

            cell_x = center_x + cell_radii * np.cos(cell_angles)
            cell_y = center_y + cell_radii * np.sin(cell_angles)

            # Create spatially structured cell types for interesting neighborhood analysis
            # Cells near center are one type, outer ring another, with some mixing
            cell_types = []
            for cx, cy in zip(cell_x, cell_y):
                dist_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                angle = np.arctan2(cy - center_y, cx - center_x)

                # Create spatial structure
                if dist_from_center < radius * 0.3:
                    # Inner core - mostly tumor cells
                    cell_types.append(np.random.choice([0, 1], p=[0.8, 0.2]))
                elif dist_from_center < radius * 0.6:
                    # Middle zone - mixed immune infiltration
                    cell_types.append(np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3]))
                else:
                    # Outer zone - stromal and immune
                    if angle > 0:
                        cell_types.append(np.random.choice([3, 4, 5], p=[0.4, 0.3, 0.3]))
                    else:
                        cell_types.append(np.random.choice([4, 5, 6], p=[0.3, 0.4, 0.3]))

            cells = pd.DataFrame({
                'label': range(n_cells_per_section),
                'cell_x': cell_x,
                'cell_y': cell_y,
                'total_counts': np.random.poisson(50, n_cells_per_section),
                'n_genes_by_counts': np.random.poisson(20, n_cells_per_section),
                'nuclei_area': np.random.uniform(100, 500, n_cells_per_section),
                'nuclei_expanded_area': np.random.uniform(200, 800, n_cells_per_section),
                'cell_type': cell_types  # Add explicit cell type
            })
            cells.to_csv(cell_dir / 'cell_metadata.csv.gz', index=False, compression='gzip')

            # Add cluster annotations matching cell types
            clusters = pd.DataFrame({
                'leiden_0.5': cell_types
            })
            clusters.to_csv(cell_dir / 'clustering_umap.csv.gz', index=False, compression='gzip')

            section_dirs.append(section_dir)
            session['status'].progress = 10 + (i + 1) * 10

        session['status'].message = 'Processing demo data...'

        # Now process the demo data
        reconstructor = G4X3DReconstructor(
            section_dirs=[str(d) for d in section_dirs],
            section_thickness=5.0,
            pixel_size=0.5
        )

        session['status'].progress = 60
        reconstructor.load_sections(verbose=False)

        session['status'].progress = 70
        reconstructor.align_sections(method='centroid', verbose=False)

        session['status'].progress = 80
        reconstructor.build_3d_volume(verbose=False)

        session['status'].progress = 90
        try:
            reconstructor.find_3d_niches(n_neighbors=15, verbose=False)
        except Exception:
            pass  # Non-fatal

        # Save results
        reconstructor.save_results(str(session['results_dir']), formats=['parquet'], verbose=False)

        session['reconstructor'] = reconstructor
        session['status'].status = 'complete'
        session['status'].progress = 100
        session['status'].message = 'Demo data loaded successfully!'
        session['status'].has_results = True
        session['status'].n_sections = n_sections
        session['status'].n_transcripts = len(reconstructor.transcripts_3d) if reconstructor.transcripts_3d is not None else 0
        session['status'].n_cells = len(reconstructor.cells_3d) if reconstructor.cells_3d is not None else 0

        return jsonify({
            'success': True,
            'n_sections': n_sections,
            'n_transcripts': session['status'].n_transcripts,
            'n_cells': session['status'].n_cells
        })

    except Exception as e:
        session['status'].status = 'error'
        session['status'].error = str(e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/session/<session_id>/reset', methods=['POST'])
def reset_session(session_id: str):
    """Reset a session and clear all data."""
    if session_id in sessions:
        session = sessions[session_id]
        # Clean up directories
        if session['upload_dir'].exists():
            shutil.rmtree(session['upload_dir'])
        if session['results_dir'].exists():
            shutil.rmtree(session['results_dir'])
        del sessions[session_id]

    return jsonify({'success': True})


# Create templates directory structure
TEMPLATES_DIR = Path(__file__).parent / 'templates'
TEMPLATES_DIR.mkdir(exist_ok=True)

STATIC_DIR = Path(__file__).parent / 'static'
STATIC_DIR.mkdir(exist_ok=True)


if __name__ == '__main__':
    print("=" * 60)
    print("G4X 3D Reconstruction Web Interface")
    print("=" * 60)
    print(f"\nStarting server at http://localhost:5000")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Results folder: {app.config['RESULTS_FOLDER']}")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=True)
