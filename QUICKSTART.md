# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (optional, defaults provided)
export CLICKHOUSE_HOST=localhost
export CLICKHOUSE_PORT=9000
export CLICKHOUSE_DATABASE=roworks
export S3_BUCKET=roworks-data
export OPENX_DATA_PATH=/data/openx
export BRIDGEDATA_PATH=/data/bridgedata
export LOG_DIR=/data/visualization/logs  # Log directory (default)
```

## Usage

### Run ETL Pipelines

```bash
# Process Open-X data
python main.py etl-openx

# Process Bridgedata
python main.py etl-bridgedata

# Run both ETL pipelines
python main.py all
```

### Generate Analysis

```bash
# Compute metrics and KPIs
python main.py analyze
```

### Create Visualizations

```bash
# Generate all dashboards
python main.py visualize
```

### Complete Workflow

```bash
# Run everything: ETL → Analysis → Visualization
python main.py all
```

## Programmatic Usage

```python
from pipelines.openx_etl import OpenXETL
from pipelines.bridgedata_etl import BridgedataETL
from analysis.metrics import AnalysisLayer
from visualization.dashboards import create_dashboards

# Run ETL
openx_etl = OpenXETL()
openx_etl.process()

bridgedata_etl = BridgedataETL()
bridgedata_etl.process()

# Analyze
analyzer = AnalysisLayer()
metrics = analyzer.get_all_metrics()

# Visualize
dashboards = create_dashboards()
```

## Output

- **ClickHouse**: Sequences stored in `unified_sequences` table
- **S3**: Preview frames, GLB assets, and sequence data
- **Dashboards**: PNG files in `DASHBOARD_OUTPUT_DIR` (default: `/data/visualization/dashboards`)
- **Logs**: Rotating log files in `LOG_DIR` (default: `/data/visualization/logs`)
  - Detailed logs in files (DEBUG+)
  - Summary logs to console (INFO+)

## Dashboard Files

1. `sequence_volume_comparison.png` - Stacked bar chart
2. `steps_distribution.png` - Histogram
3. `environments_cells.png` - Pie charts
4. `activity_frequency.png` - Horizontal bar chart
5. `source_contribution.png` - Line chart
6. `scene_density.png` - Scatter plot
7. `assets_extracted.png` - Bar chart

