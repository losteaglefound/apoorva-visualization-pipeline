"""Visualization generator for comprehensive image analysis reports"""

import os
import io
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import seaborn as sns
    import numpy as np
    from PIL import Image
except ImportError:
    plt = None
    sns = None
    np = None
    Image = None

from analysis.image_analyzer import ImageAnalyzer
from config.settings import settings
from config.logging_config import get_visualization_logger


class ImageAnalysisVisualizer:
    """Generate comprehensive image analysis visualizations"""
    
    def __init__(self):
        self.logger = get_visualization_logger()
        self.analyzer = ImageAnalyzer()
        
        if plt is None:
            error_msg = "matplotlib, seaborn, numpy, Pillow required. Install with: pip install matplotlib seaborn numpy Pillow"
            self.logger.error(error_msg)
            raise ImportError(error_msg)
        
        self.output_dir = Path(settings.DASHBOARD_OUTPUT_DIR) / "image_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (16, 10)
        self.logger.info(f"Initialized Image Analysis Visualizer. Output directory: {self.output_dir}")
    
    def create_analysis_report(self, image_path: Optional[str] = None,
                              image_data: Optional[bytes] = None,
                              s3_key: Optional[str] = None,
                              output_filename: Optional[str] = None) -> Optional[str]:
        """
        Create comprehensive image analysis report with:
        - Original image
        - Grayscale conversion
        - Edge detection
        - RGB histogram (overlaid)
        - Individual channel histograms (Red, Green, Blue)
        - Edge detection statistics
        
        Args:
            image_path: Local path to image
            image_data: Raw image bytes
            s3_key: S3 key to download image
            output_filename: Optional output filename (without extension)
            
        Returns:
            Path to saved analysis report image
        """
        self.logger.info(f"Creating image analysis report: {image_path or s3_key or 'from bytes'}")
        
        # Analyze image
        analysis = self.analyzer.analyze_image(
            image_path=image_path,
            image_data=image_data,
            s3_key=s3_key
        )
        
        if not analysis.get("image_info"):
            self.logger.error("Failed to analyze image")
            return None
        
        # Load image for display
        img = self._load_image_for_display(image_path, image_data, s3_key)
        if img is None:
            self.logger.error("Failed to load image for display")
            return None
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img)
        
        # Compute grayscale
        grayscale = self.analyzer._compute_grayscale(img_array)
        
        # Compute edge detection
        edges = self.analyzer._compute_edge_detection(img_array, method="canny")
        
        # Create figure with 2x4 grid
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
        
        # 1. Original Image (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold', pad=10)
        ax1.axis('off')
        
        # Add image info text
        info_text = f"Size: {analysis['image_info']['width']}×{analysis['image_info']['height']}\n"
        info_text += f"Pixels: {analysis['image_info']['total_pixels']:,}"
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Grayscale Image (Top Middle)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(grayscale, cmap='gray')
        ax2.set_title('Grayscale', fontsize=14, fontweight='bold', pad=10)
        ax2.axis('off')
        
        # Add grayscale statistics
        gs_stats = analysis['grayscale']['statistics']
        stats_text = f"Mean: {gs_stats['mean']:.1f}\n"
        stats_text += f"Std: {gs_stats['std']:.1f}\n"
        stats_text += f"Min: {gs_stats['min']}\n"
        stats_text += f"Max: {gs_stats['max']}"
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. Edge Detection (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(edges, cmap='gray')
        ax3.set_title('Edge Detection', fontsize=14, fontweight='bold', pad=10)
        ax3.axis('off')
        
        # Add edge detection statistics
        edge_stats = analysis.get('edge_detection', {}).get('statistics', {})
        if edge_stats:
            edge_text = f"Edge Density: {edge_stats.get('edge_density_percent', 0):.2f}%\n"
            edge_text += f"Strong Edges: {edge_stats.get('strong_edges', 0):,}\n"
            edge_text += f"Weak Edges: {edge_stats.get('weak_edges', 0):,}"
            ax3.text(0.02, 0.98, edge_text, transform=ax3.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. RGB Histogram (Top Right) - All channels overlaid
        ax4 = fig.add_subplot(gs[0, 3])
        rgb_hist = analysis['rgb_histogram']
        pixel_values = np.arange(0, 256)
        
        ax4.plot(pixel_values, rgb_hist['red'], color='salmon', label='Red', linewidth=1.5, alpha=0.8)
        ax4.plot(pixel_values, rgb_hist['green'], color='lightgreen', label='Green', linewidth=1.5, alpha=0.8)
        ax4.plot(pixel_values, rgb_hist['blue'], color='lightblue', label='Blue', linewidth=1.5, alpha=0.8)
        
        ax4.set_xlabel('Pixel Value', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax4.set_title('RGB Histogram', fontsize=14, fontweight='bold')
        ax4.set_xlim(0, 255)
        ax4.legend(fontsize=10)
        ax4.grid(alpha=0.3)
        
        # 5. Red Channel Histogram (Bottom Left)
        ax5 = fig.add_subplot(gs[1, 0])
        red_hist_data = analysis['histogram_data']['red_histogram']
        red_stats = analysis['channels']['red']
        
        ax5.bar(pixel_values, red_hist_data['frequencies'], color='salmon', alpha=0.7, edgecolor='darkred', linewidth=0.5)
        ax5.axvline(red_stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {red_stats['mean']:.1f}")
        
        ax5.set_xlabel('Pixel Value', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax5.set_title('Red Channel Histogram', fontsize=14, fontweight='bold')
        ax5.set_xlim(0, 255)
        ax5.legend(fontsize=10)
        ax5.grid(alpha=0.3, axis='y')
        
        # Add statistics text
        red_text = f"Mean: {red_stats['mean']:.1f}\n"
        red_text += f"Std: {red_stats['std']:.1f}"
        ax5.text(0.98, 0.98, red_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 6. Green Channel Histogram (Bottom Middle)
        ax6 = fig.add_subplot(gs[1, 1])
        green_hist_data = analysis['histogram_data']['green_histogram']
        green_stats = analysis['channels']['green']
        
        ax6.bar(pixel_values, green_hist_data['frequencies'], color='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=0.5)
        ax6.axvline(green_stats['mean'], color='green', linestyle='--', linewidth=2, label=f"Mean: {green_stats['mean']:.1f}")
        
        ax6.set_xlabel('Pixel Value', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax6.set_title('Green Channel Histogram', fontsize=14, fontweight='bold')
        ax6.set_xlim(0, 255)
        ax6.legend(fontsize=10)
        ax6.grid(alpha=0.3, axis='y')
        
        # Add statistics text
        green_text = f"Mean: {green_stats['mean']:.1f}\n"
        green_text += f"Std: {green_stats['std']:.1f}"
        ax6.text(0.98, 0.98, green_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 7. Blue Channel Histogram (Bottom Right)
        ax7 = fig.add_subplot(gs[1, 2])
        blue_hist_data = analysis['histogram_data']['blue_histogram']
        blue_stats = analysis['channels']['blue']
        
        ax7.bar(pixel_values, blue_hist_data['frequencies'], color='lightblue', alpha=0.7, edgecolor='darkblue', linewidth=0.5)
        ax7.axvline(blue_stats['mean'], color='blue', linestyle='--', linewidth=2, label=f"Mean: {blue_stats['mean']:.1f}")
        
        ax7.set_xlabel('Pixel Value', fontsize=12, fontweight='bold')
        ax7.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax7.set_title('Blue Channel Histogram', fontsize=14, fontweight='bold')
        ax7.set_xlim(0, 255)
        ax7.legend(fontsize=10)
        ax7.grid(alpha=0.3, axis='y')
        
        # Add statistics text
        blue_text = f"Mean: {blue_stats['mean']:.1f}\n"
        blue_text += f"Std: {blue_stats['std']:.1f}"
        ax7.text(0.98, 0.98, blue_text, transform=ax7.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 8. Edge Detection Statistics (Bottom Right)
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.axis('off')
        
        if edge_stats:
            stats_text = "Edge Detection Statistics\n" + "="*30 + "\n\n"
            stats_text += f"Edge Density: {edge_stats.get('edge_density_percent', 0):.2f}%\n"
            stats_text += f"Total Edge Pixels: {edge_stats.get('edge_pixels', 0):,}\n"
            stats_text += f"Strong Edges: {edge_stats.get('strong_edges', 0):,}\n"
            stats_text += f"Weak Edges: {edge_stats.get('weak_edges', 0):,}\n"
            stats_text += f"\nMean Edge Strength: {edge_stats.get('mean_edge_strength', 0):.1f}\n"
            stats_text += f"Std Edge Strength: {edge_stats.get('std_edge_strength', 0):.1f}\n"
            
            ax8.text(0.1, 0.9, stats_text, transform=ax8.transAxes,
                    fontsize=11, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Overall title
        overall_title = "Comprehensive Image Analysis"
        if output_filename:
            overall_title += f" - {output_filename}"
        fig.suptitle(overall_title, fontsize=16, fontweight='bold', y=0.995)
        
        # Save figure
        if output_filename:
            output_path = self.output_dir / f"{output_filename}.png"
        else:
            # Generate filename from source
            if image_path:
                filename = Path(image_path).stem
            elif s3_key:
                filename = Path(s3_key).stem
            else:
                filename = "image_analysis"
            output_path = self.output_dir / f"{filename}_analysis.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Saved image analysis report: {output_path}")
        return str(output_path)
    
    def _load_image_for_display(self, image_path: Optional[str] = None,
                                image_data: Optional[bytes] = None,
                                s3_key: Optional[str] = None) -> Optional[Image.Image]:
        """Load image for display purposes"""
        try:
            if image_data:
                return Image.open(io.BytesIO(image_data))
            elif image_path and os.path.exists(image_path):
                return Image.open(image_path)
            elif s3_key:
                return self.analyzer._download_from_s3(s3_key)
            return None
        except Exception as e:
            self.logger.error(f"Error loading image for display: {e}", exc_info=True)
            return None
    
    def analyze_sequence_images(self, sequence_id: str, source: str = "openx",
                                max_images: int = 10) -> Dict[str, str]:
        """
        Analyze images from a sequence and generate reports for each.
        
        Args:
            sequence_id: Sequence ID to analyze
            source: Data source (openx or roworks)
            max_images: Maximum number of images to analyze
            
        Returns:
            Dictionary mapping image identifiers to analysis report paths
        """
        self.logger.info(f"Analyzing images for sequence {sequence_id} from {source}")
        
        # Find images for this sequence
        # This would need to query ClickHouse or scan S3
        # For now, we'll provide a framework
        
        reports = {}
        
        # Example: Look for preview frames in S3
        prefix = f"previews/{source}/{sequence_id}/"
        s3_keys = self.analyzer.s3.list_objects(prefix)
        
        # Limit to max_images
        s3_keys = sorted(s3_keys)[:max_images]
        
        for s3_key in s3_keys:
            step_id = self._extract_step_id_from_key(s3_key)
            output_filename = f"{sequence_id}_step_{step_id:06d}"
            
            report_path = self.create_analysis_report(
                s3_key=s3_key,
                output_filename=output_filename
            )
            
            if report_path:
                reports[s3_key] = report_path
        
        self.logger.info(f"Generated {len(reports)} image analysis reports for sequence {sequence_id}")
        return reports
    
    def _extract_step_id_from_key(self, s3_key: str) -> int:
        """Extract step ID from S3 key"""
        try:
            # Expected format: previews/{source}/{sequence_id}/step_{step_id:06d}.{ext}
            parts = s3_key.split('/')
            filename = parts[-1]
            if 'step_' in filename:
                step_part = filename.split('step_')[1].split('.')[0]
                return int(step_part)
        except:
            pass
        return 0
    
    def analyze_directory_images(self, directory_path: str, 
                                max_images: int = 10,
                                output_prefix: Optional[str] = None) -> Dict[str, str]:
        """
        Analyze images from a local directory.
        
        Args:
            directory_path: Path to directory containing images
            max_images: Maximum number of images to analyze
            output_prefix: Optional prefix for output filenames
            
        Returns:
            Dictionary mapping image paths to analysis report paths
        """
        self.logger.info(f"Analyzing images from directory: {directory_path}")
        
        directory = Path(directory_path)
        if not directory.exists():
            self.logger.error(f"Directory does not exist: {directory_path}")
            return {}
        
        # Find image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
        
        # Sort and limit
        # image_files = sorted(image_files)[:max_images]
        
        reports = {}
        for img_path in image_files:
            try:
                filename = img_path.stem
                if output_prefix:
                    output_filename = f"{output_prefix}_{filename}"
                else:
                    output_filename = filename
                
                report_path = self.create_analysis_report(
                    image_path=str(img_path),
                    output_filename=output_filename
                )
                
                if report_path:
                    reports[str(img_path)] = report_path
            except Exception as e:
                self.logger.warning(f"Failed to analyze image {img_path}: {e}")
        
        self.logger.info(f"Generated {len(reports)} image analysis reports from directory")
        return reports

