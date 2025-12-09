"""Comprehensive image analysis module for detailed image analysis"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import statistics

try:
    import numpy as np
    from PIL import Image
    import io
except ImportError:
    np = None
    Image = None
    io = None

from config.logging_config import get_analysis_logger
from storage.s3_client import S3Client


class ImageAnalyzer:
    """Comprehensive image analyzer for detailed image analysis"""
    
    def __init__(self):
        self.logger = get_analysis_logger()
        self.s3 = S3Client()
        
        if np is None or Image is None:
            error_msg = "numpy and Pillow required. Install with: pip install numpy Pillow"
            self.logger.error(error_msg)
            raise ImportError(error_msg)
        
        self.logger.info("Initialized Image Analyzer")
    
    def analyze_image(self, image_path: Optional[str] = None, 
                     image_data: Optional[bytes] = None,
                     s3_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on an image.
        
        Args:
            image_path: Local path to image file
            image_data: Raw image bytes
            s3_key: S3 key to download image from
            
        Returns:
            Dictionary containing comprehensive image analysis
        """
        self.logger.debug(f"Analyzing image: path={image_path}, s3_key={s3_key}")
        
        # Load image
        img = self._load_image(image_path, image_data, s3_key)
        if img is None:
            return self._empty_analysis()
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # Extract channels
        red_channel = img_array[:, :, 0]
        green_channel = img_array[:, :, 1]
        blue_channel = img_array[:, :, 2]
        
        # Compute grayscale
        grayscale = self._compute_grayscale(img_array)
        
        # Compute edge detection
        edges = self._compute_edge_detection(img_array, method="canny")
        
        # Analyze each channel
        red_stats = self._analyze_channel(red_channel, "Red")
        green_stats = self._analyze_channel(green_channel, "Green")
        blue_stats = self._analyze_channel(blue_channel, "Blue")
        grayscale_stats = self._analyze_channel(grayscale, "Grayscale")
        
        # Analyze edge detection
        edge_stats = self._analyze_edge_detection(edges)
        
        # Compute RGB histogram (overlaid)
        rgb_histogram = self._compute_rgb_histogram(red_channel, green_channel, blue_channel)
        
        # Overall image statistics
        overall_stats = self._compute_overall_stats(img_array, grayscale)
        
        # Color distribution analysis
        color_analysis = self._analyze_color_distribution(img_array)
        
        analysis = {
            "image_info": {
                "width": width,
                "height": height,
                "total_pixels": width * height,
                "mode": img.mode,
                "format": img.format if hasattr(img, 'format') else "Unknown"
            },
            "grayscale": {
                "array": grayscale.tolist(),  # For serialization, can be removed if not needed
                "statistics": grayscale_stats
            },
            "channels": {
                "red": red_stats,
                "green": green_stats,
                "blue": blue_stats
            },
            "rgb_histogram": rgb_histogram,
            "overall_statistics": overall_stats,
            "color_distribution": color_analysis,
            "edge_detection": {
                "edges": edges.tolist(),  # For serialization, can be removed if not needed
                "statistics": edge_stats
            },
            "histogram_data": {
                "red_histogram": self._compute_histogram_data(red_channel),
                "green_histogram": self._compute_histogram_data(green_channel),
                "blue_histogram": self._compute_histogram_data(blue_channel),
                "grayscale_histogram": self._compute_histogram_data(grayscale)
            }
        }
        
        self.logger.debug(f"Completed image analysis: {width}x{height}")
        return analysis
    
    def _load_image(self, image_path: Optional[str] = None,
                   image_data: Optional[bytes] = None,
                   s3_key: Optional[str] = None) -> Optional[Image.Image]:
        """Load image from various sources"""
        try:
            if image_data:
                return Image.open(io.BytesIO(image_data))
            elif image_path and os.path.exists(image_path):
                return Image.open(image_path)
            elif s3_key:
                # Download from S3
                return self._download_from_s3(s3_key)
            else:
                self.logger.warning("No valid image source provided")
                return None
        except Exception as e:
            self.logger.error(f"Error loading image: {e}", exc_info=True)
            return None
    
    def _download_from_s3(self, s3_key: str) -> Optional[Image.Image]:
        """Download image from S3 and return PIL Image"""
        try:
            self.logger.debug(f"Downloading image from S3: {s3_key}")
            response = self.s3.s3.get_object(Bucket=self.s3.bucket, Key=s3_key)
            image_data = response['Body'].read()
            return Image.open(io.BytesIO(image_data))
        except Exception as e:
            self.logger.error(f"Error downloading from S3: {e}", exc_info=True)
            return None
    
    def _compute_grayscale(self, img_array: np.ndarray) -> np.ndarray:
        """Convert RGB image to grayscale using standard formula"""
        # Standard grayscale conversion: 0.299*R + 0.587*G + 0.114*B
        grayscale = (0.299 * img_array[:, :, 0] + 
                    0.587 * img_array[:, :, 1] + 
                    0.114 * img_array[:, :, 2]).astype(np.uint8)
        return grayscale
    
    def _compute_edge_detection(self, img_array: np.ndarray, 
                                method: str = "canny") -> np.ndarray:
        """
        Compute edge detection on image using Sobel operators (Canny-like).
        
        Args:
            img_array: RGB image array
            method: Edge detection method ('canny' or 'sobel')
            
        Returns:
            Binary edge map (0 = no edge, 255 = edge)
        """
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            grayscale = self._compute_grayscale(img_array).astype(np.float32)
        else:
            grayscale = img_array.astype(np.float32)
        
        if method == "canny":
            # Canny-like edge detection using Sobel operators
            # Step 1: Apply Gaussian blur (simplified with box filter)
            kernel_size = 5
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
            blurred = self._convolve2d(grayscale, kernel)
            
            # Step 2: Compute gradients using Sobel operators
            sobel_x = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=np.float32)
            
            sobel_y = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]], dtype=np.float32)
            
            grad_x = self._convolve2d(blurred, sobel_x)
            grad_y = self._convolve2d(blurred, sobel_y)
            
            # Step 3: Compute gradient magnitude
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Step 4: Apply thresholding (adaptive)
            # Use percentile-based thresholds
            low_threshold = np.percentile(gradient_magnitude, 20)
            high_threshold = np.percentile(gradient_magnitude, 80)
            
            # Create binary edge map
            edges = np.zeros_like(gradient_magnitude)
            edges[gradient_magnitude > high_threshold] = 255
            edges[(gradient_magnitude > low_threshold) & 
                  (gradient_magnitude <= high_threshold)] = 128  # Weak edges
            
            # Step 5: Non-maximum suppression (simplified)
            edges = self._non_maximum_suppression(gradient_magnitude, grad_x, grad_y, edges)
            
            return edges.astype(np.uint8)
        
        elif method == "sobel":
            # Simple Sobel edge detection
            sobel_x = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=np.float32)
            
            sobel_y = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]], dtype=np.float32)
            
            grad_x = self._convolve2d(grayscale, sobel_x)
            grad_y = self._convolve2d(grayscale, sobel_y)
            
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Normalize to 0-255
            gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
            
            # Apply threshold
            threshold = np.percentile(gradient_magnitude, 70)
            edges = (gradient_magnitude > threshold).astype(np.uint8) * 255
            
            return edges
        
        else:
            raise ValueError(f"Unknown edge detection method: {method}")
    
    def _convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Perform 2D convolution using vectorized operations"""
        kernel_height, kernel_width = kernel.shape
        image_height, image_width = image.shape
        
        # Pad image
        pad_h = kernel_height // 2
        pad_w = kernel_width // 2
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        
        # Vectorized convolution using numpy
        output = np.zeros_like(image, dtype=np.float32)
        
        # Use numpy's vectorized operations for faster convolution
        for i in range(image_height):
            for j in range(image_width):
                output[i, j] = np.sum(padded[i:i+kernel_height, j:j+kernel_width] * kernel)
        
        return output
    
    def _non_maximum_suppression(self, magnitude: np.ndarray, 
                                 grad_x: np.ndarray, grad_y: np.ndarray,
                                 edges: np.ndarray) -> np.ndarray:
        """Simplified non-maximum suppression"""
        height, width = magnitude.shape
        suppressed = edges.copy()
        
        # Compute gradient direction
        angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
        angle[angle < 0] += 180
        
        # Suppress non-maximum pixels
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if edges[i, j] > 0:
                    # Determine neighbors based on gradient direction
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
                    elif 22.5 <= angle[i, j] < 67.5:
                        neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
                    elif 67.5 <= angle[i, j] < 112.5:
                        neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
                    else:  # 112.5 <= angle < 157.5
                        neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
                    
                    if magnitude[i, j] < max(neighbors):
                        suppressed[i, j] = 0
        
        return suppressed
    
    def _analyze_channel(self, channel: np.ndarray, channel_name: str) -> Dict[str, Any]:
        """Analyze a single color channel"""
        flat_channel = channel.flatten()
        
        return {
            "channel_name": channel_name,
            "mean": float(np.mean(flat_channel)),
            "median": float(np.median(flat_channel)),
            "std": float(np.std(flat_channel)),
            "min": int(np.min(flat_channel)),
            "max": int(np.max(flat_channel)),
            "percentiles": {
                "p25": float(np.percentile(flat_channel, 25)),
                "p50": float(np.percentile(flat_channel, 50)),
                "p75": float(np.percentile(flat_channel, 75)),
                "p90": float(np.percentile(flat_channel, 90)),
                "p95": float(np.percentile(flat_channel, 95))
            }
        }
    
    def _compute_rgb_histogram(self, red: np.ndarray, green: np.ndarray, 
                              blue: np.ndarray, bins: int = 256) -> Dict[str, Any]:
        """Compute RGB histogram with all channels overlaid"""
        red_hist, _ = np.histogram(red.flatten(), bins=bins, range=(0, 256))
        green_hist, _ = np.histogram(green.flatten(), bins=bins, range=(0, 256))
        blue_hist, _ = np.histogram(blue.flatten(), bins=bins, range=(0, 256))
        
        return {
            "red": red_hist.tolist(),
            "green": green_hist.tolist(),
            "blue": blue_hist.tolist(),
            "bins": bins,
            "range": [0, 255]
        }
    
    def _compute_histogram_data(self, channel: np.ndarray, bins: int = 256) -> Dict[str, Any]:
        """Compute histogram data for a channel"""
        hist, bin_edges = np.histogram(channel.flatten(), bins=bins, range=(0, 256))
        
        return {
            "frequencies": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
            "max_frequency": int(np.max(hist)),
            "bins": bins
        }
    
    def _compute_overall_stats(self, img_array: np.ndarray, 
                               grayscale: np.ndarray) -> Dict[str, Any]:
        """Compute overall image statistics"""
        # Brightness (mean of grayscale)
        brightness = float(np.mean(grayscale))
        
        # Contrast (standard deviation of grayscale)
        contrast = float(np.std(grayscale))
        
        # Colorfulness (variance of color channels)
        color_variance = float(np.var(img_array))
        
        # Dominant color channel
        channel_means = [
            float(np.mean(img_array[:, :, 0])),  # Red
            float(np.mean(img_array[:, :, 1])),  # Green
            float(np.mean(img_array[:, :, 2]))   # Blue
        ]
        dominant_channel = ["Red", "Green", "Blue"][np.argmax(channel_means)]
        
        return {
            "brightness": brightness,
            "contrast": contrast,
            "color_variance": color_variance,
            "dominant_channel": dominant_channel,
            "channel_means": {
                "red": channel_means[0],
                "green": channel_means[1],
                "blue": channel_means[2]
            }
        }
    
    def _analyze_color_distribution(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze color distribution in the image"""
        # Count unique colors (approximate by quantizing)
        # For performance, sample pixels
        sample_size = min(10000, img_array.size // 3)
        indices = np.random.choice(img_array.size // 3, sample_size, replace=False)
        sampled_pixels = img_array.reshape(-1, 3)[indices]
        
        # Quantize to reduce unique colors
        quantized = (sampled_pixels // 32) * 32  # Quantize to 8 levels per channel
        unique_colors = len(np.unique(quantized.view(np.dtype((np.void, quantized.dtype.itemsize*3)))))
        
        # Color saturation analysis
        max_channel = np.max(img_array, axis=2)
        min_channel = np.min(img_array, axis=2)
        # Avoid division by zero
        saturation = np.where(max_channel > 0, 
                             (max_channel - min_channel) / np.maximum(max_channel, 1), 
                             0)
        avg_saturation = float(np.mean(saturation))
        
        return {
            "unique_colors_estimate": int(unique_colors),
            "average_saturation": avg_saturation,
            "saturation_std": float(np.std(saturation))
        }
    
    def _analyze_edge_detection(self, edges: np.ndarray) -> Dict[str, Any]:
        """Analyze edge detection results"""
        # Count edge pixels (non-zero)
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.size
        edge_density = (edge_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        # Count strong edges (255) vs weak edges (128)
        strong_edges = np.sum(edges == 255)
        weak_edges = np.sum(edges == 128)
        
        # Edge statistics
        edge_values = edges[edges > 0]
        if len(edge_values) > 0:
            mean_edge_strength = float(np.mean(edge_values))
            std_edge_strength = float(np.std(edge_values))
        else:
            mean_edge_strength = 0.0
            std_edge_strength = 0.0
        
        return {
            "edge_pixels": int(edge_pixels),
            "total_pixels": int(total_pixels),
            "edge_density_percent": float(edge_density),
            "strong_edges": int(strong_edges),
            "weak_edges": int(weak_edges),
            "mean_edge_strength": mean_edge_strength,
            "std_edge_strength": std_edge_strength
        }
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure"""
        return {
            "image_info": {},
            "grayscale": {"statistics": {}},
            "channels": {"red": {}, "green": {}, "blue": {}},
            "rgb_histogram": {},
            "overall_statistics": {},
            "color_distribution": {},
            "edge_detection": {"statistics": {}},
            "histogram_data": {}
        }

