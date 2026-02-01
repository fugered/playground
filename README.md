# G4X 3D Reconstruction Pipeline

A comprehensive Python pipeline for 3D reconstruction of spatial transcriptomic and proteomic data from serial sections acquired on the **Singular Genomics G4X** platform.

## Overview

This pipeline enables researchers to:

- **Load** data from multiple serial sections
- **Align** sections across the z-stack (registration)
- **Build** unified 3D coordinate spaces for transcripts and cells
- **Identify** 3D cellular niches and microenvironments
- **Track** cells across adjacent sections
- **Visualize** results in interactive 3D plots
- **Export** for downstream tools (napari, Vitessce, scanpy)

## G4X Output File Structure

The pipeline expects the standard G4X output directory structure:

```
section_XX/
├── rna/
│   ├── transcript_table.csv.gz     # x_pixel_coordinate, y_pixel_coordinate, gene_name, cell_id
│   └── raw_features.parquet        # Raw features for re-demultiplexing
├── single_cell_data/
│   ├── cell_metadata.csv.gz        # cell_x, cell_y, label, total_counts, etc.
│   ├── cell_by_transcript.csv.gz   # Cell × Gene count matrix
│   ├── cell_by_protein.csv.gz      # Cell × Protein intensity matrix (multiomics)
│   ├── clustering_umap.csv.gz      # Pre-computed clusters and UMAP
│   ├── dgex.csv.gz                 # Differential expression results
│   └── feature_matrix.h5           # AnnData-compatible format
├── segmentation/
│   └── segmentation_mask.npz       # Arrays: 'nuclei', 'nuclei_exp'
├── protein/                        # (multiomics only)
│   ├── {protein}.jp2               # Full-resolution protein images
│   └── {protein}_thumbnail.png
├── h_and_e/
│   ├── h_and_e.jp2                 # Fluorescent H&E
│   ├── nuclear.jp2
│   └── cytoplasmic.jp2
├── metrics/
│   ├── transcript_core_metrics.csv
│   └── per_area_metrics.csv
└── g4x_viewer/
    ├── {sample_id}_HE.ome.tiff
    ├── {sample_id}_multiplex.ome.tiff
    └── {sample_id}_segmentation.bin
```

## Installation

```bash
# Install dependencies:
pip install -r requirements.txt

# For full functionality (optional):
pip install scanpy anndata napari
```

## Web Interface

A browser-based interface is available for interactive data import and visualization.

### Starting the Web Interface

```bash
# Simple launch (opens browser automatically)
python run_web.py

# Custom port
python run_web.py --port 8080

# Production mode (accessible from network)
python run_web.py --host 0.0.0.0 --port 5000
```

### Web Interface Features

- **File Upload**: Drag & drop CSV, Parquet, or ZIP files for each section
- **Processing Parameters**: Configure section thickness, pixel size, alignment method
- **Interactive 3D Visualization**: Explore transcripts and cells in 3D using Plotly
- **Gene Expression**: Search and visualize individual gene expression patterns
- **Niche Analysis**: View identified 3D cellular niches
- **Export**: Download processed results in Parquet format

### Demo Mode

Click "Load Demo" to generate synthetic data and explore the interface without uploading real data.

## Quick Start (Python API)

```python
from g4x_3d_reconstruction import G4X3DReconstructor

# Define paths to your 9 serial sections (in z-order!)
section_dirs = [
    "/path/to/section_01",
    "/path/to/section_02",
    # ... up to section_09
]

# Initialize
reconstructor = G4X3DReconstructor(
    section_dirs=section_dirs,
    section_thickness=5.0,  # µm (standard FFPE)
    pixel_size=0.5          # µm (G4X default)
)

# Load data
reconstructor.load_sections()

# Align sections
reconstructor.align_sections(method='centroid')

# Build 3D volume
reconstructor.build_3d_volume()

# Find 3D niches
reconstructor.find_3d_niches(n_neighbors=30)

# Save results
reconstructor.save_results("/output/path")
```

## Key Features

### 1. Section Alignment

Two registration methods are available:

- **Centroid alignment** (`method='centroid'`): Fast, aligns by tissue center of mass
- **ICP alignment** (`method='icp'`): Iterative Closest Point on cell centroids for better accuracy

```python
reconstructor.align_sections(
    method='icp',
    reference_index=4,    # Use middle section as reference
    max_shift=500.0       # Maximum allowed shift in pixels
)
```

### 2. 3D Niche Identification

Cells are clustered based on their 3D neighborhood composition:

```python
reconstructor.find_3d_niches(
    n_neighbors=30,       # Cells in neighborhood
    resolution=0.5,       # Leiden clustering resolution
    use_protein=True      # Include protein data
)
```

### 3. Cell Tracking

Link cells across adjacent sections based on spatial proximity:

```python
cell_tracks = reconstructor.track_cells_across_sections(
    max_distance=50.0     # Max distance for matching (pixels)
)
```

### 4. Interactive Visualization

```python
import g4x_3d_visualization as viz

# 3D transcript scatter
fig = viz.plot_transcripts_3d(
    reconstructor.transcripts_3d,
    color_by='gene_name',
    genes=['CD3E', 'CD8A', 'CD4']
)
fig.show()

# 3D cell scatter with niches
fig = viz.plot_cells_3d(
    reconstructor.cells_3d,
    color_by='niche_3d'
)
fig.show()

# Full dashboard
viz.create_dashboard(reconstructor, "dashboard.html")
```

### 5. Export Options

```python
# Tabular data (Parquet/CSV)
reconstructor.save_results("/output", formats=['parquet', 'csv'])

# For napari 3D viewer
reconstructor.export_for_napari("/output/napari")

# For Vitessce web viewer
reconstructor.export_for_vitessce("/output/vitessce")
```

## Output Files

After running the pipeline:

```
output/
├── transcripts_3d.parquet      # All transcripts with 3D coordinates
├── transcripts_3d.csv.gz
├── cells_3d.parquet            # All cells with 3D coordinates + niches
├── cells_3d.csv.gz
├── alignment_transforms.json   # Section alignment matrices
├── napari/
│   ├── transcripts_3d.parquet
│   ├── cells_3d.parquet
│   └── load_napari.py          # Script to load in napari
├── vitessce/
│   ├── cells_3d.h5ad
│   └── vitessce_config.json
└── g4x_3d_dashboard.html       # Interactive HTML dashboard
```

## Data Schema

### 3D Transcripts Table (`transcripts_3d.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `x` | float | X coordinate (pixels, aligned) |
| `y` | float | Y coordinate (pixels, aligned) |
| `z` | float | Z coordinate (µm) |
| `gene_name` | str | Gene symbol |
| `probe_name` | str | Probe identifier |
| `confidence_score` | float | Detection confidence |
| `cell_id` | int | Cell assignment (0 = unassigned) |
| `cell_id_global` | str | Global cell ID (section_cellid) |
| `section_id` | str | Source section identifier |
| `section_index` | int | Section z-index (0-8) |

### 3D Cells Table (`cells_3d.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `cell_x` | float | Cell centroid X (pixels, aligned) |
| `cell_y` | float | Cell centroid Y (pixels, aligned) |
| `cell_z` | float | Cell centroid Z (µm) |
| `label` | int | Original cell ID from section |
| `cell_id_global` | str | Global unique cell ID |
| `total_counts` | int | Total transcript counts |
| `n_genes_by_counts` | int | Number of unique genes |
| `nuclei_area` | int | Nuclear area (pixels²) |
| `nuclei_expanded_area` | int | Expanded cell area (pixels²) |
| `niche_3d` | int | 3D niche cluster assignment |
| `section_id` | str | Source section identifier |
| `section_index` | int | Section z-index (0-8) |

## Physical Coordinates

To convert pixel coordinates to physical units (µm):

```python
pixel_size = 0.5  # µm per pixel (check your run metadata)

x_um = x_pixels * pixel_size
y_um = y_pixels * pixel_size
# z is already in µm
```

## Tips for Best Results

1. **Section Order**: Ensure sections are listed in correct z-order (bottom to top)

2. **Alignment Quality**: Check alignment visually before proceeding. Large shifts may indicate sample damage or incorrect ordering.

3. **Niche Resolution**: Start with `resolution=0.5` and adjust based on your tissue complexity

4. **Memory Management**: For very large datasets, the pipeline automatically subsamples for visualization

5. **Protein Data**: If running multiomics, protein intensities are included in cell metadata

## Troubleshooting

### "File not found" errors
- Verify your section directory paths
- Check that the G4X pipeline completed successfully for all sections

### Poor alignment
- Try `method='icp'` for more accurate registration
- Consider using image-based registration on H&E images
- Check if any sections have significantly different tissue coverage

### Memory issues
- Process fewer sections at a time
- Use Parquet format (more efficient than CSV)
- Subsample transcripts for visualization

## Files Included

- `g4x_3d_reconstruction.py` - Main reconstruction pipeline
- `g4x_3d_visualization.py` - Interactive visualization functions
- `g4x_3d_reconstruction_notebook.ipynb` - Jupyter notebook tutorial
- `app.py` - Flask web application backend
- `run_web.py` - Web interface launcher script
- `templates/index.html` - Web interface frontend
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## License

MIT License
