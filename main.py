"""Main entry point for RoWorks data analysis and visualization"""

import argparse
from pathlib import Path

from pipelines.openx_etl import OpenXETL
from pipelines.bridgedata_etl import BridgedataETL
from analysis.metrics import AnalysisLayer
from visualization.dashboards import create_dashboards
from visualization.image_analysis_visualizer import ImageAnalysisVisualizer
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
        choices=["etl-openx", "etl-bridgedata", "analyze", "visualize", "analyze-image", "all"],
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
    parser.add_argument(
        "--image-path",
        type=str,
        help="Path to image file for analysis (used with analyze-image command)"
    )
    parser.add_argument(
        "--s3-key",
        type=str,
        help="S3 key for image to analyze (used with analyze-image command)"
    )
    parser.add_argument(
        "--sequence-id",
        type=str,
        help="Sequence ID to analyze images from (used with analyze-image command)"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["openx", "roworks"],
        default="openx",
        help="Data source for sequence analysis (used with analyze-image command)"
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
                if name == "image_analyses" and isinstance(path, dict):
                    logger.info(f"  {name}: {len(path)} image analysis reports")
                    for img_key, report_path in list(path.items())[:5]:  # Show first 5
                        logger.info(f"    {img_key}: {report_path}")
                    if len(path) > 5:
                        logger.info(f"    ... and {len(path) - 5} more")
                elif path:
                    logger.info(f"  {name}: {path}")
                else:
                    logger.warning(f"  {name}: Failed to generate")
        except Exception as e:
            logger.error(f"Visualization failed: {e}", exc_info=True)
            raise
    
    if args.command == "analyze-image":
        logger.info("Running Image Analysis")
        try:
            visualizer = ImageAnalysisVisualizer()
            
            if args.sequence_id:
                # Analyze all images from a sequence
                logger.info(f"Analyzing images for sequence: {args.sequence_id} (source: {args.source})")
                reports = visualizer.analyze_sequence_images(
                    sequence_id=args.sequence_id,
                    source=args.source,
                    max_images=10
                )
                logger.info(f"Generated {len(reports)} image analysis reports:")
                for s3_key, report_path in reports.items():
                    logger.info(f"  {s3_key}: {report_path}")
            elif args.image_path:
                # Analyze single image from local path
                logger.info(f"Analyzing image: {args.image_path}")
                report_path = visualizer.create_analysis_report(image_path=args.image_path)
                if report_path:
                    logger.info(f"Image analysis report saved: {report_path}")
                else:
                    logger.error("Failed to generate image analysis report")
            elif args.s3_key:
                # Analyze single image from S3
                logger.info(f"Analyzing image from S3: {args.s3_key}")
                report_path = visualizer.create_analysis_report(s3_key=args.s3_key)
                if report_path:
                    logger.info(f"Image analysis report saved: {report_path}")
                else:
                    logger.error("Failed to generate image analysis report")
            else:
                logger.error("Please provide --image-path, --s3-key, or --sequence-id for image analysis")
        except Exception as e:
            logger.error(f"Image analysis failed: {e}", exc_info=True)
            raise
    
    logger.info("=" * 60)
    logger.info("All operations completed successfully")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

