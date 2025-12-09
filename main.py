"""Main entry point for RoWorks data analysis and visualization"""

import argparse
from pathlib import Path

from pipelines.openx_etl import OpenXETL
from pipelines.bridgedata_etl import BridgedataETL
from analysis.metrics import AnalysisLayer
from visualization.dashboards import create_dashboards
from config.settings import settings
from config.logging_config import get_main_logger


def main():
    logger = get_main_logger()
    logger.info("=" * 60)
    logger.info("RoWorks Data Analysis & Visualization")
    logger.info("=" * 60)
    parser = argparse.ArgumentParser(description="RoWorks Data Analysis & Visualization")
    parser.add_argument(
        "command",
        choices=["etl-openx", "etl-bridgedata", "analyze", "visualize", "all"],
        help="Command to execute"
    )
    parser.add_argument(
        "--openx-path",
        type=str,
        default=settings.OPENX_DATA_PATH,
        help="Path to Open-X data"
    )
    parser.add_argument(
        "--bridgedata-path",
        type=str,
        default=settings.BRIDGEDATA_PATH,
        help="Path to Bridgedata"
    )
    
    args = parser.parse_args()
    
    if args.command == "etl-openx" or args.command == "all":
        logger.info("Running Open-X ETL Pipeline")
        try:
            etl = OpenXETL()
            result = etl.process()
            logger.info(f"Open-X ETL completed: {result}")
        except Exception as e:
            logger.error(f"Open-X ETL failed: {e}", exc_info=True)
            raise
    
    if args.command == "etl-bridgedata" or args.command == "all":
        logger.info("Running Bridgedata ETL Pipeline")
        try:
            etl = BridgedataETL()
            result = etl.process()
            logger.info(f"Bridgedata ETL completed: {result}")
        except Exception as e:
            logger.error(f"Bridgedata ETL failed: {e}", exc_info=True)
            raise
    
    if args.command == "analyze" or args.command == "all":
        logger.info("Running Analysis Layer")
        try:
            analyzer = AnalysisLayer()
            metrics = analyzer.get_all_metrics()
            
            # Log summary
            logger.info("Global Metrics:")
            logger.info(f"  Total Sequences: {metrics['global_metrics']['total_sequences']}")
            logger.info(f"  Open-X Sequences: {metrics['global_metrics']['openx_sequences']}")
            logger.info(f"  RoWorks Sequences: {metrics['global_metrics']['roworks_sequences']}")
            logger.info(f"  Unique Environments: {metrics['global_metrics']['unique_environments']}")
            logger.info(f"  Unique Cells: {metrics['global_metrics']['unique_cells']}")
            logger.info(f"  Unique Activities: {metrics['global_metrics']['unique_activities']}")
            
            logger.info("Performance Indicators:")
            pi = metrics['performance_indicators']
            logger.info(f"  Usable for Training: {pi['usable_for_training']['percentage']:.1f}%")
            logger.info(f"  Avg Activity Complexity: {pi['activity_complexity']['avg_total_complexity']:.2f}")
            
            logger.info("Data KPIs:")
            kpis = metrics['data_kpis']
            logger.info(f"  Unique RoWorks Environments: {kpis['unique_roworks_environments']}")
            logger.info(f"  OEM Robot-Model Pairs: {kpis['oem_robot_model_pairs']}")
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            raise
    
    if args.command == "visualize" or args.command == "all":
        logger.info("Generating Visualization Dashboards")
        try:
            dashboards = create_dashboards()
            logger.info("Generated Dashboards:")
            for name, path in dashboards.items():
                if path:
                    logger.info(f"  {name}: {path}")
                else:
                    logger.warning(f"  {name}: Failed to generate")
        except Exception as e:
            logger.error(f"Visualization failed: {e}", exc_info=True)
            raise
    
    logger.info("=" * 60)
    logger.info("All operations completed successfully")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

