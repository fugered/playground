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
from g4x_3d_visualization import (
    plot_transcripts_3d,
    plot_cells_3d,
    plot_section_comparison,
    plot_gene_expression_3d,
    plot_niche_distribution
)

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

            cells = pd.DataFrame({
                'label': range(n_cells_per_section),
                'cell_x': cell_x,
                'cell_y': cell_y,
                'total_counts': np.random.poisson(50, n_cells_per_section),
                'n_genes_by_counts': np.random.poisson(20, n_cells_per_section),
                'nuclei_area': np.random.uniform(100, 500, n_cells_per_section),
                'nuclei_expanded_area': np.random.uniform(200, 800, n_cells_per_section)
            })
            cells.to_csv(cell_dir / 'cell_metadata.csv.gz', index=False, compression='gzip')

            # Add cluster annotations
            clusters = pd.DataFrame({
                'leiden_0.5': np.random.randint(0, 8, n_cells_per_section)
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
