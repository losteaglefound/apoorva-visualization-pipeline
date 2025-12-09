# RoWorks Data Analysis & Visualization Guide

This project provides a unified system for analyzing and visualizing Open-X and Bridgedata to feed into:
- **RoWorks Model Training Hub**
- **RoWorks Analyzer (VLA-I)**
- **Data Dashboards**

## Project Structure

```
/data/visualization/
├── schema/
│   └── unified_schema.py          # Unified activity schema
├── pipelines/
│   ├── openx_etl.py               # Open-X ETL pipeline (Pipeline A)
│   └── bridgedata_etl.py          # Bridgedata ETL pipeline (Pipeline B)
├── analysis/
│   └── metrics.py                 # Analysis layer with global metrics and KPIs
├── visualization/
│   └── dashboards.py              # Visualization dashboards
├── storage/
│   ├── clickhouse_client.py       # ClickHouse integration
│   └── s3_client.py               # S3 storage utilities
├── utils/
│   └── helpers.py                 # Utility functions
├── config/
│   ├── settings.py                # Configuration settings
│   └── logging_config.py          # Logging configuration with rotating handlers
└── logs/                          # Log files directory (created automatically)
    ├── openx_etl.log
    ├── bridgedata_etl.log
    ├── analysis.log
    ├── visualization.log
    ├── clickhouse.log
    ├── s3.log
    └── main.log
```

## Data Types

### Open-X Data
- Preexisting activities & robot-human sequences
- Rich labels (activity_key, sequence_length, environment, object taxonomy)
- Inconsistent directory structures → normalized by Pipeline A

### Bridgedata (RoWorks / BridgeData V2)
- Raw trajectory data with JPEG images (640x480 resolution)
- Natural language annotations for each trajectory
- 60,096 trajectories across 24 environments and 13 skills
- Teleoperated demonstrations and scripted rollouts
- Multiple camera views (over-the-shoulder, randomized, depth, wrist)
- Control frequency: 5 Hz, average trajectory length: 38 timesteps
- Normalized into Open-X-compatible schema by Pipeline B

## Unified Schema

The `UnifiedSequence` schema includes:
- **Required fields**: sequence_id, source, activity_key, environment_key, cell_key, num_steps
- **Objects**: objects_used (with GLB paths)
- **Robot**: robot_model
- **Human**: human_actions
- **Metadata**: payload, cycle_time, etc.

## Processing Pipelines

### Pipeline A — Open-X ETL
1. Read Open-X JSON index
2. Normalize paths
3. Extract metadata (sequence_length, env_key, objects, activity)
4. Store + index into ClickHouse
5. Store preview frames into S3 → link in DB

### Pipeline B — BridgeData V2 ETL
1. Find trajectory directories containing image files (JPEGs)
2. Parse trajectory images and NumPy files (if present)
3. Extract natural language annotations from files
4. Infer environment and activity from path/annotations
5. Generate unified sequences with step frames
6. Upload images to S3 as preview frames
7. Store sequences in ClickHouse

## Analysis Layer

### Global Metrics
- # of sequences (Open-X vs Bridgedata)
- # of steps per sequence distribution
- # of environments
- # of cells
- # of unique activities

### Performance Indicators
- % of sequences usable for model training
- Avg. frame resolution
- Avg. activity complexity (# objects, # transitions)
- Spatial density (from LiDAR → point cloud spread)

### Data KPIs
- Real factory → synthetic sequence conversion rate
- Unique environment templates derived from RoWorks
- # of OEM-ready robot-model pairs covered

## Visualization Dashboards

1. **Sequence Volume Comparison** (Stacked Bar)
2. **Steps per Sequence Distribution** (Histogram)
3. **Environments & Cells** (Pie Chart)
4. **Activity Key Frequency** (Horizontal Bar)
5. **Source Contribution Over Time** (Line Chart)
6. **3D Scene Density** (Scatter)
7. **3D Assets Extracted** (Bar)

## Installation

```bash
pip install -r requirements.txt
```

## Logging

The system uses rotating file handlers for detailed logging and console handlers for concise terminal output:

- **File Logs**: Detailed DEBUG+ logs saved to `/data/visualization/logs/` (configurable via `LOG_DIR` env var)
  - Each service has its own log file (e.g., `openx_etl.log`, `bridgedata_etl.log`)
  - Logs rotate when they reach 10MB (configurable)
  - Keeps 5 backup files (configurable)
  - Includes: timestamp, logger name, level, filename, line number, function name, message

- **Console Logs**: INFO+ level logs to terminal
  - Simplified format: timestamp, level, message
  - Less verbose for better readability

### Log Files

- `openx_etl.log` - Open-X ETL pipeline logs
- `bridgedata_etl.log` - Bridgedata ETL pipeline logs
- `analysis.log` - Analysis layer logs
- `visualization.log` - Dashboard generation logs
- `clickhouse.log` - ClickHouse client logs
- `s3.log` - S3 client logs
- `main.log` - Main application logs

### Configuration

Set `LOG_DIR` environment variable to change log directory:
```bash
export LOG_DIR=/path/to/logs
```

## Usage

```python
# Run Open-X ETL
from pipelines.openx_etl import OpenXETL
etl = OpenXETL()
etl.process()

# Run Bridgedata ETL
from pipelines.bridgedata_etl import BridgedataETL
etl = BridgedataETL()
etl.process()

# Generate analysis
from analysis.metrics import AnalysisLayer
analyzer = AnalysisLayer()
metrics = analyzer.compute_global_metrics()

# Visualize
from visualization.dashboards import create_dashboards
create_dashboards(metrics)
```

