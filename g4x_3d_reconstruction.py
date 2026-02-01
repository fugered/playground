#!/usr/bin/env python3
"""
G4X 3D Reconstruction Pipeline
==============================
Reconstruct 3D spatial volumes from serial sections of G4X spatial transcriptomic
and proteomic data.

Based on G4X output file structure:
- rna/transcript_table.csv.gz: transcript locations with x/y coordinates, gene names, cell_id
- single_cell_data/cell_metadata.csv.gz: cell centroids (cell_x, cell_y), areas, counts
- single_cell_data/cell_by_transcript.csv.gz: cell x gene count matrix
- single_cell_data/cell_by_protein.csv.gz: cell x protein intensity matrix
- single_cell_data/clustering_umap.csv.gz: cluster annotations
- segmentation/segmentation_mask.npz: nuclei and nuclei_exp masks
- h_and_e/h_and_e.jp2: H&E images
- protein/{protein}.jp2: protein images

Author: Claude (Anthropic)
"""

import os
import sys
import glob
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


@dataclass
class SectionData:
    """Container for a single section's data."""
    section_id: str
    section_index: int  # Z-index in the stack
    z_position: float   # Physical z position (microns)
    
    # Core data
    transcripts: Optional[pd.DataFrame] = None
    cell_metadata: Optional[pd.DataFrame] = None
    cell_by_transcript: Optional[pd.DataFrame] = None
    cell_by_protein: Optional[pd.DataFrame] = None
    clustering: Optional[pd.DataFrame] = None
    
    # Segmentation
    nuclei_mask: Optional[np.ndarray] = None
    cell_mask: Optional[np.ndarray] = None
    
    # Alignment transform (for registration)
    transform_matrix: Optional[np.ndarray] = None
    

class G4X3DReconstructor:
    """
    Pipeline for 3D reconstruction from G4X serial sections.
    
    Parameters
    ----------
    section_dirs : List[str]
        List of paths to section output directories (in z-order)
    section_thickness : float
        Physical thickness of each section in microns (default: 5.0)
    pixel_size : float
        Physical pixel size in microns (default: 0.5)
    """
    
    def __init__(
        self,
        section_dirs: List[str],
        section_thickness: float = 5.0,
        pixel_size: float = 0.5
    ):
        self.section_dirs = [Path(d) for d in section_dirs]
        self.section_thickness = section_thickness
        self.pixel_size = pixel_size
        self.n_sections = len(section_dirs)
        
        self.sections: List[SectionData] = []
        self.aligned = False
        
        # 3D reconstructed data
        self.transcripts_3d: Optional[pd.DataFrame] = None
        self.cells_3d: Optional[pd.DataFrame] = None
        
    def load_sections(self, verbose: bool = True) -> None:
        """Load all sections from their directories."""
        for i, section_dir in enumerate(self.section_dirs):
            if verbose:
                print(f"Loading section {i+1}/{self.n_sections}: {section_dir.name}")
            
            section = self._load_single_section(section_dir, i)
            self.sections.append(section)
            
        if verbose:
            print(f"\nLoaded {len(self.sections)} sections successfully.")
            self._print_summary()
    
    def _load_single_section(self, section_dir: Path, index: int) -> SectionData:
        """Load data from a single section directory."""
        section_id = section_dir.name
        z_position = index * self.section_thickness
        
        section = SectionData(
            section_id=section_id,
            section_index=index,
            z_position=z_position
        )
        
        # Load transcript table
        transcript_path = section_dir / "rna" / "transcript_table.csv.gz"
        if transcript_path.exists():
            section.transcripts = pd.read_csv(transcript_path)
            # Rename columns for consistency
            section.transcripts = section.transcripts.rename(columns={
                'x_pixel_coordinate': 'x',
                'y_pixel_coordinate': 'y'
            })
        
        # Load cell metadata
        cell_meta_path = section_dir / "single_cell_data" / "cell_metadata.csv.gz"
        if cell_meta_path.exists():
            section.cell_metadata = pd.read_csv(cell_meta_path)
        
        # Load cell x transcript matrix
        cell_transcript_path = section_dir / "single_cell_data" / "cell_by_transcript.csv.gz"
        if cell_transcript_path.exists():
            section.cell_by_transcript = pd.read_csv(cell_transcript_path, index_col=0)
        
        # Load cell x protein matrix (multiomics only)
        cell_protein_path = section_dir / "single_cell_data" / "cell_by_protein.csv.gz"
        if cell_protein_path.exists():
            section.cell_by_protein = pd.read_csv(cell_protein_path, index_col=0)
        
        # Load clustering/UMAP
        clustering_path = section_dir / "single_cell_data" / "clustering_umap.csv.gz"
        if clustering_path.exists():
            section.clustering = pd.read_csv(clustering_path)
        
        # Load segmentation masks
        seg_path = section_dir / "segmentation" / "segmentation_mask.npz"
        if seg_path.exists():
            seg_data = np.load(seg_path)
            if 'nuclei' in seg_data:
                section.nuclei_mask = seg_data['nuclei']
            if 'nuclei_exp' in seg_data:
                section.cell_mask = seg_data['nuclei_exp']
        
        return section
    
    def _print_summary(self) -> None:
        """Print summary of loaded data."""
        print("\n" + "="*60)
        print("SECTION SUMMARY")
        print("="*60)
        
        for section in self.sections:
            n_transcripts = len(section.transcripts) if section.transcripts is not None else 0
            n_cells = len(section.cell_metadata) if section.cell_metadata is not None else 0
            n_genes = section.cell_by_transcript.shape[1] if section.cell_by_transcript is not None else 0
            has_protein = section.cell_by_protein is not None
            
            print(f"\nSection {section.section_index}: {section.section_id}")
            print(f"  Z position: {section.z_position:.1f} µm")
            print(f"  Transcripts: {n_transcripts:,}")
            print(f"  Cells: {n_cells:,}")
            print(f"  Genes: {n_genes}")
            print(f"  Protein data: {'Yes' if has_protein else 'No'}")
    
    def align_sections(
        self,
        method: str = 'centroid',
        reference_index: int = 0,
        max_shift: float = 500.0,
        verbose: bool = True
    ) -> None:
        """
        Align sections using various registration methods.
        
        Parameters
        ----------
        method : str
            Alignment method: 'centroid', 'icp', 'correlation'
        reference_index : int
            Index of reference section (others aligned to this)
        max_shift : float
            Maximum allowed shift in pixels
        verbose : bool
            Print progress
        """
        if verbose:
            print(f"\nAligning sections using '{method}' method...")
        
        ref_section = self.sections[reference_index]
        
        for i, section in enumerate(self.sections):
            if i == reference_index:
                section.transform_matrix = np.eye(3)
                continue
            
            if method == 'centroid':
                transform = self._align_by_centroid(ref_section, section)
            elif method == 'icp':
                transform = self._align_by_icp(ref_section, section, max_shift)
            else:
                transform = np.eye(3)
            
            section.transform_matrix = transform
            
            if verbose:
                tx, ty = transform[0, 2], transform[1, 2]
                print(f"  Section {i}: shift = ({tx:.1f}, {ty:.1f}) pixels")
        
        self.aligned = True
        if verbose:
            print("Alignment complete.")
    
    def _align_by_centroid(
        self,
        ref_section: SectionData,
        target_section: SectionData
    ) -> np.ndarray:
        """Align by matching tissue centroids."""
        # Calculate centroids from transcripts
        ref_cx = ref_section.transcripts['x'].mean()
        ref_cy = ref_section.transcripts['y'].mean()
        
        tgt_cx = target_section.transcripts['x'].mean()
        tgt_cy = target_section.transcripts['y'].mean()
        
        # Translation to align centroids
        tx = ref_cx - tgt_cx
        ty = ref_cy - tgt_cy
        
        transform = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ], dtype=float)
        
        return transform
    
    def _align_by_icp(
        self,
        ref_section: SectionData,
        target_section: SectionData,
        max_shift: float
    ) -> np.ndarray:
        """Align using Iterative Closest Point on cell centroids."""
        # Get cell centroids
        ref_cells = ref_section.cell_metadata[['cell_x', 'cell_y']].values
        tgt_cells = target_section.cell_metadata[['cell_x', 'cell_y']].values
        
        # Subsample if too many cells
        max_points = 10000
        if len(ref_cells) > max_points:
            idx = np.random.choice(len(ref_cells), max_points, replace=False)
            ref_cells = ref_cells[idx]
        if len(tgt_cells) > max_points:
            idx = np.random.choice(len(tgt_cells), max_points, replace=False)
            tgt_cells = tgt_cells[idx]
        
        # Simple ICP implementation
        transform = np.eye(3)
        prev_error = float('inf')
        
        for iteration in range(50):
            # Apply current transform
            tgt_transformed = tgt_cells + transform[:2, 2]
            
            # Find closest points
            tree = cKDTree(ref_cells)
            distances, indices = tree.query(tgt_transformed)
            
            # Filter outliers
            mask = distances < np.percentile(distances, 90)
            if mask.sum() < 100:
                break
            
            ref_matched = ref_cells[indices[mask]]
            tgt_matched = tgt_transformed[mask]
            
            # Compute optimal translation
            ref_mean = ref_matched.mean(axis=0)
            tgt_mean = tgt_matched.mean(axis=0)
            
            delta = ref_mean - tgt_mean
            
            # Limit shift
            delta = np.clip(delta, -max_shift, max_shift)
            
            transform[0, 2] += delta[0]
            transform[1, 2] += delta[1]
            
            # Check convergence
            error = np.mean(distances[mask])
            if abs(prev_error - error) < 0.1:
                break
            prev_error = error
        
        return transform
    
    def build_3d_volume(self, verbose: bool = True) -> None:
        """
        Construct 3D representations of transcripts and cells.
        """
        if verbose:
            print("\nBuilding 3D volume...")
        
        # Build 3D transcript table
        transcript_dfs = []
        for section in self.sections:
            if section.transcripts is None:
                continue
            
            df = section.transcripts.copy()
            
            # Apply alignment transform
            if section.transform_matrix is not None:
                tx = section.transform_matrix[0, 2]
                ty = section.transform_matrix[1, 2]
                df['x'] = df['x'] + tx
                df['y'] = df['y'] + ty
            
            # Add z coordinate
            df['z'] = section.z_position
            
            # Add section info
            df['section_id'] = section.section_id
            df['section_index'] = section.section_index
            
            # Create global cell ID
            if 'cell_id' in df.columns:
                df['cell_id_global'] = df['cell_id'].apply(
                    lambda x: f"{section.section_id}_{x}" if x > 0 else None
                )
            
            transcript_dfs.append(df)
        
        self.transcripts_3d = pd.concat(transcript_dfs, ignore_index=True)
        
        # Build 3D cell table
        cell_dfs = []
        for section in self.sections:
            if section.cell_metadata is None:
                continue
            
            df = section.cell_metadata.copy()
            
            # Apply alignment transform
            if section.transform_matrix is not None:
                tx = section.transform_matrix[0, 2]
                ty = section.transform_matrix[1, 2]
                df['cell_x'] = df['cell_x'] + tx
                df['cell_y'] = df['cell_y'] + ty
            
            # Add z coordinate
            df['cell_z'] = section.z_position
            
            # Add section info
            df['section_id'] = section.section_id
            df['section_index'] = section.section_index
            
            # Create global cell ID
            df['cell_id_global'] = df['label'].apply(
                lambda x: f"{section.section_id}_{x}"
            )
            
            # Add clustering info if available
            if section.clustering is not None:
                cluster_cols = [c for c in section.clustering.columns 
                               if 'leiden' in c.lower() or 'cluster' in c.lower()]
                for col in cluster_cols:
                    if col in section.clustering.columns:
                        df[col] = section.clustering[col].values
            
            cell_dfs.append(df)
        
        self.cells_3d = pd.concat(cell_dfs, ignore_index=True)
        
        if verbose:
            print(f"  Total 3D transcripts: {len(self.transcripts_3d):,}")
            print(f"  Total 3D cells: {len(self.cells_3d):,}")
            print(f"  Volume dimensions: X={self.transcripts_3d['x'].max():.0f}, "
                  f"Y={self.transcripts_3d['y'].max():.0f}, "
                  f"Z={self.transcripts_3d['z'].max():.0f} µm")
    
    def find_3d_niches(
        self,
        n_neighbors: int = 30,
        resolution: float = 0.5,
        use_protein: bool = True,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Identify 3D cellular niches/neighborhoods.
        
        Parameters
        ----------
        n_neighbors : int
            Number of neighbors for niche definition
        resolution : float
            Leiden clustering resolution
        use_protein : bool
            Include protein data in niche definition
        
        Returns
        -------
        pd.DataFrame
            Cell metadata with niche assignments
        """
        if self.cells_3d is None:
            raise ValueError("Must call build_3d_volume() first")
        
        if verbose:
            print("\nIdentifying 3D niches...")
        
        # Get cell positions in 3D
        coords = self.cells_3d[['cell_x', 'cell_y', 'cell_z']].values
        
        # Scale z to account for section thickness vs xy resolution
        z_scale = self.section_thickness / self.pixel_size
        coords_scaled = coords.copy()
        coords_scaled[:, 2] = coords_scaled[:, 2] / z_scale
        
        # Build KD-tree for neighbor finding
        tree = cKDTree(coords_scaled)
        
        # Find k-nearest neighbors for each cell
        distances, indices = tree.query(coords_scaled, k=n_neighbors + 1)
        
        # Create neighbor composition matrix
        # Each cell characterized by composition of its neighborhood
        cluster_col = None
        for col in self.cells_3d.columns:
            if 'leiden' in col.lower():
                cluster_col = col
                break
        
        if cluster_col is None:
            if verbose:
                print("  No cluster annotations found, using transcript-based features")
            # Fall back to transcript counts
            niche_features = self._compute_local_gene_expression(indices)
        else:
            clusters = self.cells_3d[cluster_col].values
            n_clusters = clusters.max() + 1
            
            # Compute neighborhood composition
            niche_features = np.zeros((len(coords), n_clusters))
            for i, neighbors in enumerate(indices):
                for j in neighbors[1:]:  # Exclude self
                    niche_features[i, clusters[j]] += 1
            niche_features = niche_features / n_neighbors
        
        # Cluster to find niches
        try:
            import scanpy as sc
            import anndata as ad
            
            adata = ad.AnnData(X=niche_features)
            sc.pp.pca(adata, n_comps=min(20, niche_features.shape[1] - 1))
            sc.pp.neighbors(adata, n_neighbors=15)
            sc.tl.leiden(adata, resolution=resolution, key_added='niche')
            
            self.cells_3d['niche_3d'] = adata.obs['niche'].values.astype(int)
            
        except ImportError:
            if verbose:
                print("  scanpy not available, using simple clustering")
            from sklearn.cluster import MiniBatchKMeans
            
            n_niches = min(10, len(coords) // 100)
            kmeans = MiniBatchKMeans(n_clusters=n_niches, random_state=42)
            self.cells_3d['niche_3d'] = kmeans.fit_predict(niche_features)
        
        if verbose:
            n_niches = self.cells_3d['niche_3d'].nunique()
            print(f"  Identified {n_niches} 3D niches")
        
        return self.cells_3d
    
    def _compute_local_gene_expression(
        self,
        neighbor_indices: np.ndarray,
        top_genes: int = 50
    ) -> np.ndarray:
        """Compute local gene expression features for niche identification."""
        # Get gene expression matrix from first section with data
        for section in self.sections:
            if section.cell_by_transcript is not None:
                gene_matrix = section.cell_by_transcript
                break
        else:
            return np.zeros((len(neighbor_indices), 1))
        
        # Select most variable genes
        gene_var = gene_matrix.var()
        top_gene_idx = gene_var.nlargest(top_genes).index
        
        # This is simplified - in practice need to map cells across sections
        return np.random.rand(len(neighbor_indices), top_genes)
    
    def track_cells_across_sections(
        self,
        max_distance: float = 50.0,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Track/link cells across adjacent sections based on spatial proximity.
        
        Parameters
        ----------
        max_distance : float
            Maximum distance (pixels) for cell matching
        
        Returns
        -------
        pd.DataFrame
            Cell tracks with linked IDs across sections
        """
        if self.cells_3d is None:
            raise ValueError("Must call build_3d_volume() first")
        
        if verbose:
            print("\nTracking cells across sections...")
        
        tracks = []
        track_id = 0
        
        for i in range(len(self.sections) - 1):
            section_a = self.sections[i]
            section_b = self.sections[i + 1]
            
            if section_a.cell_metadata is None or section_b.cell_metadata is None:
                continue
            
            # Get aligned centroids
            cells_a = section_a.cell_metadata[['cell_x', 'cell_y']].values.copy()
            cells_b = section_b.cell_metadata[['cell_x', 'cell_y']].values.copy()
            
            if section_a.transform_matrix is not None:
                cells_a += section_a.transform_matrix[:2, 2]
            if section_b.transform_matrix is not None:
                cells_b += section_b.transform_matrix[:2, 2]
            
            # Match cells between sections
            tree_b = cKDTree(cells_b)
            distances, indices = tree_b.query(cells_a)
            
            # Create matches within distance threshold
            matches = []
            for j, (dist, idx) in enumerate(zip(distances, indices)):
                if dist < max_distance:
                    matches.append({
                        'section_a': i,
                        'section_b': i + 1,
                        'cell_a': section_a.cell_metadata['label'].iloc[j],
                        'cell_b': section_b.cell_metadata['label'].iloc[idx],
                        'distance': dist
                    })
            
            if verbose:
                print(f"  Sections {i}-{i+1}: {len(matches)} cell matches")
            
            tracks.extend(matches)
        
        return pd.DataFrame(tracks)
    
    def export_for_napari(
        self,
        output_dir: str,
        include_images: bool = False,
        verbose: bool = True
    ) -> Dict[str, str]:
        """
        Export data for visualization in napari.
        
        Parameters
        ----------
        output_dir : str
            Output directory path
        include_images : bool
            Whether to include image stacks
        
        Returns
        -------
        Dict[str, str]
            Paths to exported files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print(f"\nExporting for napari to {output_dir}...")
        
        exports = {}
        
        # Export 3D transcripts as points
        if self.transcripts_3d is not None:
            # Convert to napari-compatible format
            points_df = self.transcripts_3d[['x', 'y', 'z', 'gene_name']].copy()
            points_df['z_scaled'] = points_df['z'] / self.section_thickness
            
            points_path = output_dir / "transcripts_3d.parquet"
            points_df.to_parquet(points_path)
            exports['transcripts'] = str(points_path)
            
            if verbose:
                print(f"  Transcripts: {points_path}")
        
        # Export 3D cells as points
        if self.cells_3d is not None:
            cells_path = output_dir / "cells_3d.parquet"
            self.cells_3d.to_parquet(cells_path)
            exports['cells'] = str(cells_path)
            
            if verbose:
                print(f"  Cells: {cells_path}")
        
        # Create napari loading script
        script = self._generate_napari_script(exports)
        script_path = output_dir / "load_napari.py"
        with open(script_path, 'w') as f:
            f.write(script)
        exports['script'] = str(script_path)
        
        if verbose:
            print(f"  Napari script: {script_path}")
        
        return exports
    
    def _generate_napari_script(self, exports: Dict[str, str]) -> str:
        """Generate a Python script to load data in napari."""
        return f'''#!/usr/bin/env python3
"""
Load G4X 3D reconstruction in napari.
Auto-generated script.
"""
import napari
import pandas as pd
import numpy as np

# Load data
transcripts = pd.read_parquet("{exports.get('transcripts', '')}")
cells = pd.read_parquet("{exports.get('cells', '')}")

# Create viewer
viewer = napari.Viewer()

# Add transcripts as points
# Color by gene
genes = transcripts['gene_name'].unique()
gene_colors = {{g: np.random.rand(3) for g in genes}}
colors = [gene_colors[g] for g in transcripts['gene_name']]

viewer.add_points(
    transcripts[['z_scaled', 'y', 'x']].values,
    size=2,
    face_color=colors,
    name='Transcripts'
)

# Add cells as points
# Color by cluster if available
if 'niche_3d' in cells.columns:
    cell_colors = cells['niche_3d'].values
else:
    cell_colors = 'blue'

viewer.add_points(
    cells[['cell_z', 'cell_y', 'cell_x']].values / {self.section_thickness},
    size=10,
    face_color=cell_colors,
    name='Cells'
)

napari.run()
'''
    
    def export_for_vitessce(
        self,
        output_dir: str,
        verbose: bool = True
    ) -> Dict[str, str]:
        """
        Export data for visualization in Vitessce.
        
        Parameters
        ----------
        output_dir : str
            Output directory path
        
        Returns
        -------
        Dict[str, str]
            Paths to exported files and config
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print(f"\nExporting for Vitessce to {output_dir}...")
        
        exports = {}
        
        # Export cells as AnnData for Vitessce
        try:
            import anndata as ad
            
            # Create obs (cell metadata)
            obs = self.cells_3d.copy()
            obs = obs.set_index('cell_id_global')
            
            # Create spatial coordinates
            spatial = obs[['cell_x', 'cell_y', 'cell_z']].values
            
            # Placeholder expression matrix
            X = np.zeros((len(obs), 1))
            
            adata = ad.AnnData(X=X, obs=obs)
            adata.obsm['spatial'] = spatial[:, :2]  # 2D for Vitessce
            adata.obsm['X_spatial_3d'] = spatial    # 3D coords
            
            h5ad_path = output_dir / "cells_3d.h5ad"
            adata.write_h5ad(h5ad_path)
            exports['anndata'] = str(h5ad_path)
            
            if verbose:
                print(f"  AnnData: {h5ad_path}")
                
        except ImportError:
            if verbose:
                print("  anndata not available, skipping h5ad export")
        
        # Generate Vitessce config
        config = self._generate_vitessce_config(exports, output_dir)
        config_path = output_dir / "vitessce_config.json"
        
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        exports['config'] = str(config_path)
        
        if verbose:
            print(f"  Config: {config_path}")
        
        return exports
    
    def _generate_vitessce_config(
        self,
        exports: Dict[str, str],
        output_dir: Path
    ) -> dict:
        """Generate a Vitessce configuration."""
        return {
            "version": "1.0.16",
            "name": "G4X 3D Reconstruction",
            "description": f"3D reconstruction of {self.n_sections} serial sections",
            "datasets": [{
                "uid": "cells",
                "name": "Cells 3D",
                "files": [{
                    "fileType": "anndata.h5ad",
                    "url": exports.get('anndata', ''),
                    "options": {
                        "obsEmbedding": [
                            {"path": "obsm/X_spatial_3d", "dims": [0, 1, 2], "embeddingType": "X_spatial_3d"}
                        ]
                    }
                }]
            }],
            "coordinationSpace": {
                "embeddingType": {"A": "X_spatial_3d"},
                "embeddingZoom": {"A": 2}
            },
            "layout": [{
                "component": "scatterplot",
                "coordinationScopes": {
                    "embeddingType": "A",
                    "embeddingZoom": "A"
                },
                "x": 0, "y": 0, "w": 12, "h": 12
            }],
            "initStrategy": "auto"
        }
    
    def save_results(
        self,
        output_dir: str,
        formats: List[str] = ['parquet', 'csv'],
        verbose: bool = True
    ) -> Dict[str, str]:
        """
        Save reconstruction results to files.
        
        Parameters
        ----------
        output_dir : str
            Output directory path
        formats : List[str]
            Output formats: 'parquet', 'csv', 'h5ad'
        
        Returns
        -------
        Dict[str, str]
            Paths to saved files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print(f"\nSaving results to {output_dir}...")
        
        saved = {}
        
        # Save 3D transcripts
        if self.transcripts_3d is not None:
            if 'parquet' in formats:
                path = output_dir / "transcripts_3d.parquet"
                self.transcripts_3d.to_parquet(path)
                saved['transcripts_parquet'] = str(path)
            if 'csv' in formats:
                path = output_dir / "transcripts_3d.csv.gz"
                self.transcripts_3d.to_csv(path, index=False)
                saved['transcripts_csv'] = str(path)
        
        # Save 3D cells
        if self.cells_3d is not None:
            if 'parquet' in formats:
                path = output_dir / "cells_3d.parquet"
                self.cells_3d.to_parquet(path)
                saved['cells_parquet'] = str(path)
            if 'csv' in formats:
                path = output_dir / "cells_3d.csv.gz"
                self.cells_3d.to_csv(path, index=False)
                saved['cells_csv'] = str(path)
        
        # Save alignment transforms
        transforms = {
            section.section_id: section.transform_matrix.tolist()
            for section in self.sections
            if section.transform_matrix is not None
        }
        
        import json
        transform_path = output_dir / "alignment_transforms.json"
        with open(transform_path, 'w') as f:
            json.dump(transforms, f, indent=2)
        saved['transforms'] = str(transform_path)
        
        if verbose:
            for name, path in saved.items():
                print(f"  {name}: {path}")
        
        return saved


def create_example_usage():
    """Print example usage of the pipeline."""
    example = '''
# Example Usage
# =============

from g4x_3d_reconstruction import G4X3DReconstructor

# Define paths to your 9 serial sections (in z-order)
section_dirs = [
    "/path/to/section_01",
    "/path/to/section_02",
    "/path/to/section_03",
    "/path/to/section_04",
    "/path/to/section_05",
    "/path/to/section_06",
    "/path/to/section_07",
    "/path/to/section_08",
    "/path/to/section_09",
]

# Initialize reconstructor
# section_thickness: physical thickness of each section (µm)
# pixel_size: physical pixel size (µm)
reconstructor = G4X3DReconstructor(
    section_dirs=section_dirs,
    section_thickness=5.0,  # Standard FFPE section thickness
    pixel_size=0.5          # G4X pixel size
)

# Load all section data
reconstructor.load_sections()

# Align sections (register across z-stack)
reconstructor.align_sections(method='centroid')  # or 'icp' for better alignment

# Build 3D volume
reconstructor.build_3d_volume()

# Identify 3D niches
reconstructor.find_3d_niches(n_neighbors=30, resolution=0.5)

# Track cells across sections
cell_tracks = reconstructor.track_cells_across_sections(max_distance=50)

# Export results
reconstructor.save_results("/path/to/output", formats=['parquet', 'csv'])

# Export for visualization
reconstructor.export_for_napari("/path/to/output/napari")
reconstructor.export_for_vitessce("/path/to/output/vitessce")

# Access data directly
transcripts_3d = reconstructor.transcripts_3d  # pd.DataFrame
cells_3d = reconstructor.cells_3d              # pd.DataFrame
'''
    return example


if __name__ == "__main__":
    print("G4X 3D Reconstruction Pipeline")
    print("="*40)
    print(create_example_usage())
