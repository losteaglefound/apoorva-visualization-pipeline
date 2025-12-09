"""Visualization dashboards for RoWorks data analysis"""

import os
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter
import json

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import seaborn as sns
    import pandas as pd
    import numpy as np
except ImportError:
    plt = None
    sns = None
    pd = None
    np = None

from analysis.metrics import AnalysisLayer
from config.settings import settings
from config.logging_config import get_visualization_logger


class DashboardGenerator:
    """Generate visualization dashboards"""
    
    def __init__(self):
        self.logger = get_visualization_logger()
        
        if plt is None:
            error_msg = "matplotlib, seaborn, pandas, numpy required. Install with: pip install matplotlib seaborn pandas numpy"
            self.logger.error(error_msg)
            raise ImportError(error_msg)
        
        self.logger.info("Initializing Dashboard Generator")
        self.analyzer = AnalysisLayer()
        self.output_dir = Path(settings.DASHBOARD_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Dashboard output directory: {self.output_dir}")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        self.logger.debug("Configured matplotlib style")
    
    def create_all_dashboards(self) -> Dict[str, str]:
        """Create all dashboards"""
        self.logger.info("Starting dashboard generation")
        metrics = self.analyzer.get_all_metrics()
        self.logger.debug("Retrieved metrics for visualization")
        
        dashboards = {}
        dashboard_methods = [
            ("sequence_volume", self.create_sequence_volume_comparison),
            ("steps_distribution", self.create_steps_distribution),
            ("environments_cells", self.create_environments_cells),
            ("activity_frequency", self.create_activity_frequency),
            ("source_contribution", self.create_source_contribution),
            ("scene_density", self.create_scene_density),
            ("assets_extracted", self.create_assets_extracted)
        ]
        
        for name, method in dashboard_methods:
            try:
                self.logger.info(f"Creating dashboard: {name}")
                dashboards[name] = method(metrics)
                self.logger.info(f"Successfully created {name}: {dashboards[name]}")
            except Exception as e:
                self.logger.error(f"Error creating dashboard {name}: {e}", exc_info=True)
                dashboards[name] = None
        
        self.logger.info(f"Dashboard generation complete. Created {sum(1 for v in dashboards.values() if v)} dashboards")
        return dashboards
    
    def create_sequence_volume_comparison(self, metrics: Dict[str, Any]) -> str:
        """A. Sequence Volume Comparison (Stacked Bar)"""
        global_metrics = metrics["global_metrics"]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sources = ["openx", "roworks"]
        counts = [
            global_metrics["sequences_by_source"]["openx"],
            global_metrics["sequences_by_source"]["roworks"]
        ]
        
        colors = ["#3498db", "#e74c3c"]
        bars = ax.bar(sources, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Number of Sequences', fontsize=12, fontweight='bold')
        ax.set_xlabel('Data Source', fontsize=12, fontweight='bold')
        ax.set_title('Sequence Volume Comparison: Open-X vs RoWorks', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "sequence_volume_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def create_steps_distribution(self, metrics: Dict[str, Any]) -> str:
        """B. Steps per Sequence Distribution (Histogram)"""
        global_metrics = metrics["global_metrics"]
        steps_dist = global_metrics["steps_distribution"]
        
        # Get actual data for histogram
        analyzer = AnalysisLayer()
        all_sequences = analyzer.clickhouse.query_sequences(limit=100000)
        steps = [s["num_steps"] for s in all_sequences if s.get("num_steps")]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create histogram
        n, bins, patches = ax.hist(steps, bins=50, color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add vertical lines for statistics
        ax.axvline(steps_dist["mean"], color='red', linestyle='--', linewidth=2, label=f'Mean: {steps_dist["mean"]:.1f}')
        ax.axvline(steps_dist["median"], color='green', linestyle='--', linewidth=2, label=f'Median: {steps_dist["median"]:.1f}')
        
        ax.set_xlabel('Number of Steps per Sequence', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Steps per Sequence Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "steps_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def create_environments_cells(self, metrics: Dict[str, Any]) -> str:
        """C. Environments & Cells (Pie Chart)"""
        global_metrics = metrics["global_metrics"]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Environments pie chart
        env_counts = Counter(global_metrics.get("environments_list", []))
        if env_counts:
            top_envs = dict(env_counts.most_common(10))
            other_count = sum(env_counts.values()) - sum(top_envs.values())
            if other_count > 0:
                top_envs["Other"] = other_count
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(top_envs)))
            ax1.pie(top_envs.values(), labels=top_envs.keys(), autopct='%1.1f%%',
                   colors=colors, startangle=90)
            ax1.set_title('Environments Distribution', fontsize=12, fontweight='bold')
        
        # Cells pie chart
        cell_counts = Counter(global_metrics.get("cells_list", []))
        if cell_counts:
            top_cells = dict(cell_counts.most_common(10))
            other_count = sum(cell_counts.values()) - sum(top_cells.values())
            if other_count > 0:
                top_cells["Other"] = other_count
            
            colors = plt.cm.Pastel1(np.linspace(0, 1, len(top_cells)))
            ax2.pie(top_cells.values(), labels=top_cells.keys(), autopct='%1.1f%%',
                   colors=colors, startangle=90)
            ax2.set_title('Cells Distribution', fontsize=12, fontweight='bold')
        
        plt.suptitle('Environments & Cells Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_path = self.output_dir / "environments_cells.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def create_activity_frequency(self, metrics: Dict[str, Any]) -> str:
        """D. Activity Key Frequency (Horizontal Bar)"""
        global_metrics = metrics["global_metrics"]
        
        # Get activity counts
        analyzer = AnalysisLayer()
        all_sequences = analyzer.clickhouse.query_sequences(limit=100000)
        activity_counts = Counter(s["activity_key"] for s in all_sequences if s.get("activity_key"))
        
        # Get top activities
        top_activities = dict(activity_counts.most_common(15))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        activities = list(top_activities.keys())
        counts = list(top_activities.values())
        
        y_pos = np.arange(len(activities))
        bars = ax.barh(y_pos, counts, color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{int(count)}',
                   ha='left', va='center', fontsize=10, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(activities)
        ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_ylabel('Activity Key', fontsize=12, fontweight='bold')
        ax.set_title('Activity Key Frequency (Top 15)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "activity_frequency.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def create_source_contribution(self, metrics: Dict[str, Any]) -> str:
        """E. Source Contribution Over Time (Line Chart)"""
        analyzer = AnalysisLayer()
        all_sequences = analyzer.clickhouse.query_sequences(limit=100000)
        
        # Group by date (if available) or use sequence index as proxy
        # For now, we'll create a cumulative line chart
        openx_cumulative = []
        roworks_cumulative = []
        
        openx_count = 0
        roworks_count = 0
        
        for seq in sorted(all_sequences, key=lambda x: x.get("created_at", "")):
            if seq["source"] == "openx":
                openx_count += 1
            elif seq["source"] == "roworks":
                roworks_count += 1
            
            openx_cumulative.append(openx_count)
            roworks_cumulative.append(roworks_count)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(len(all_sequences))
        ax.plot(x, openx_cumulative, label='Open-X', color='#3498db', linewidth=2, marker='o', markersize=3)
        ax.plot(x, roworks_cumulative, label='RoWorks', color='#e74c3c', linewidth=2, marker='s', markersize=3)
        
        ax.set_xlabel('Sequence Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Sequences', fontsize=12, fontweight='bold')
        ax.set_title('Source Contribution Over Time (Cumulative)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "source_contribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def create_scene_density(self, metrics: Dict[str, Any]) -> str:
        """F. 3D Scene Density (Scatter)"""
        analyzer = AnalysisLayer()
        all_sequences = analyzer.clickhouse.query_sequences(limit=100000)
        
        # Filter RoWorks sequences (they have LiDAR data)
        roworks_sequences = [s for s in all_sequences if s["source"] == "roworks"]
        
        # Use num_steps as proxy for density (more steps = more spatial coverage)
        # In reality, you'd compute actual point cloud density
        x = [s["num_steps"] for s in roworks_sequences]
        y = [s.get("num_steps", 0) / 100.0 for s in roworks_sequences]  # Density estimate
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(x, y, alpha=0.6, c=range(len(x)), cmap='viridis', s=50, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Number of Steps', fontsize=12, fontweight='bold')
        ax.set_ylabel('Spatial Density (Estimated)', fontsize=12, fontweight='bold')
        ax.set_title('3D Scene Density (RoWorks Sequences)', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sequence Index', fontsize=10)
        
        plt.tight_layout()
        output_path = self.output_dir / "scene_density.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def create_assets_extracted(self, metrics: Dict[str, Any]) -> str:
        """G. 3D Assets Extracted (Bar)"""
        analyzer = AnalysisLayer()
        all_sequences = analyzer.clickhouse.query_sequences(limit=100000)
        
        # Count GLB assets by source
        openx_assets = 0
        roworks_assets = 0
        
        for seq in all_sequences:
            try:
                objects_used = json.loads(seq.get("objects_used", "[]"))
                if isinstance(objects_used, list):
                    glb_count = sum(1 for obj in objects_used if obj.get("glb_path"))
                    if seq["source"] == "openx":
                        openx_assets += glb_count
                    elif seq["source"] == "roworks":
                        roworks_assets += glb_count
            except:
                pass
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sources = ["Open-X", "RoWorks"]
        counts = [openx_assets, roworks_assets]
        colors = ["#3498db", "#e74c3c"]
        
        bars = ax.bar(sources, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Number of GLB Assets', fontsize=12, fontweight='bold')
        ax.set_xlabel('Data Source', fontsize=12, fontweight='bold')
        ax.set_title('3D Assets Extracted', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "assets_extracted.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)


def create_dashboards() -> Dict[str, str]:
    """Convenience function to create all dashboards"""
    generator = DashboardGenerator()
    return generator.create_all_dashboards()

