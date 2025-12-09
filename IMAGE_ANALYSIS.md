# Image Analysis Feature

This document describes the comprehensive image analysis feature that provides in-depth analysis of each image in your visualization pipeline.

## Overview

The image analysis feature generates detailed reports for each image, including:

1. **Original Image** - The source image with dimensions and pixel count
2. **Grayscale Conversion** - Converted grayscale version with statistics
3. **Edge Detection** - Canny-style edge detection showing object boundaries and contours
4. **RGB Histogram** - Overlaid histogram showing Red, Green, and Blue channel distributions
5. **Individual Channel Histograms** - Separate histograms for Red, Green, and Blue channels with statistics
6. **Edge Detection Statistics** - Detailed statistics about edge density and strength

Each analysis report includes:
- Image dimensions and total pixel count
- Grayscale statistics (mean, std, min, max)
- Per-channel statistics (mean, std, percentiles)
- Color distribution analysis
- Overall image statistics (brightness, contrast, color variance)

## Usage

### Command Line

#### Analyze a Single Image from Local Path
```bash
python main.py analyze-image --image-path /path/to/image.jpg
```

#### Analyze a Single Image from S3
```bash
python main.py analyze-image --s3-key previews/openx/sequence_123/step_000001.jpg
```

#### Analyze All Images from a Sequence
```bash
python main.py analyze-image --sequence-id sequence_123 --source openx
```

#### Analyze Images from a Directory
```python
from visualization.image_analysis_visualizer import ImageAnalysisVisualizer

visualizer = ImageAnalysisVisualizer()
reports = visualizer.analyze_directory_images(
    directory_path="/path/to/images",
    max_images=10,
    output_prefix="analysis"
)
```

### Automatic Integration

When running the visualization command, image analysis is automatically performed on sample sequences:

```bash
python main.py visualize
```

This will:
1. Generate all standard dashboards
2. Automatically analyze sample images from Open-X and RoWorks sequences
3. Save analysis reports to `{DASHBOARD_OUTPUT_DIR}/image_analysis/`

## Output Format

Each image analysis report is saved as a PNG file with a 2x4 grid layout:

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ Original     │  Grayscale   │ Edge Detect  │ RGB Histogram│
│ Image        │              │              │              │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ Red Channel  │ Green Channel│ Blue Channel │ Edge Stats   │
│ Histogram    │ Histogram    │ Histogram    │              │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

## Analysis Details

### Grayscale Conversion
- Uses standard formula: `0.299*R + 0.587*G + 0.114*B`
- Provides mean, standard deviation, min, and max values

### RGB Histogram
- Shows pixel value distribution (0-255) for all three channels
- Overlaid visualization for easy comparison
- Frequency counts for each pixel value

### Channel-Specific Analysis
Each color channel (Red, Green, Blue) includes:
- Mean pixel value
- Standard deviation
- Minimum and maximum values
- Percentiles (25th, 50th, 75th, 90th, 95th)
- Histogram with frequency distribution

### Edge Detection
- **Canny-style Edge Detection**: Detects object boundaries and contours
- **Edge Density**: Percentage of pixels that are edges
- **Strong Edges**: Count of high-confidence edge pixels
- **Weak Edges**: Count of low-confidence edge pixels
- **Mean Edge Strength**: Average intensity of detected edges
- **Edge Statistics**: Standard deviation and distribution of edge strengths

### Overall Statistics
- **Brightness**: Mean grayscale value
- **Contrast**: Standard deviation of grayscale
- **Color Variance**: Variance across all color channels
- **Dominant Channel**: Channel with highest mean value
- **Color Saturation**: Average saturation across the image

## File Structure

```
/data/visualization/
├── analysis/
│   └── image_analyzer.py          # Core image analysis logic
├── visualization/
│   └── image_analysis_visualizer.py  # Visualization generator
└── dashboards/
    └── image_analysis/             # Output directory for reports
        ├── sequence_123_step_000001_analysis.png
        ├── sequence_123_step_000002_analysis.png
        └── ...
```

## API Reference

### ImageAnalyzer

Main class for performing image analysis.

```python
from analysis.image_analyzer import ImageAnalyzer

analyzer = ImageAnalyzer()
analysis = analyzer.analyze_image(
    image_path="/path/to/image.jpg",
    # OR
    s3_key="previews/openx/seq_123/step_000001.jpg",
    # OR
    image_data=bytes_data
)
```

### ImageAnalysisVisualizer

Class for generating visualization reports.

```python
from visualization.image_analysis_visualizer import ImageAnalysisVisualizer

visualizer = ImageAnalysisVisualizer()
report_path = visualizer.create_analysis_report(
    image_path="/path/to/image.jpg",
    output_filename="my_analysis"
)
```

## Integration with Current Flow

The image analysis is integrated into the existing visualization pipeline:

1. **ETL Pipelines** - Images are uploaded to S3 during processing
2. **Analysis Layer** - Metrics are computed (existing functionality)
3. **Visualization** - Dashboards are generated, and now image analysis reports are also created
4. **Output** - All reports are saved to the dashboard output directory

## Example Output

When you run `python main.py visualize`, you'll see:

```
Generated Dashboards:
  sequence_volume_comparison: /data/visualization/dashboards/sequence_volume_comparison.png
  steps_distribution: /data/visualization/dashboards/steps_distribution.png
  ...
  image_analyses: 6 image analysis reports
    previews/openx/seq_001/step_000001.jpg: /data/visualization/dashboards/image_analysis/seq_001_step_000001_analysis.png
    previews/openx/seq_001/step_000002.jpg: /data/visualization/dashboards/image_analysis/seq_001_step_000002_analysis.png
    ...
```

## Requirements

All required dependencies are already in `requirements.txt`:
- `Pillow>=10.0.0` - Image processing
- `numpy>=1.24.0` - Numerical operations
- `matplotlib>=3.7.0` - Visualization
- `seaborn>=0.12.0` - Enhanced plotting

No additional dependencies are needed.

