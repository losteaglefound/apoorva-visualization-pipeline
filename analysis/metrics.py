"""Analysis layer for computing global metrics, performance indicators, and KPIs"""

from typing import Dict, List, Any, Optional
from collections import Counter
import json
import statistics

from storage.clickhouse_client import ClickHouseClient
from config.settings import settings
from config.logging_config import get_analysis_logger


class AnalysisLayer:
    """Analysis layer for computing metrics and KPIs"""
    
    def __init__(self):
        self.logger = get_analysis_logger()
        self.clickhouse = ClickHouseClient()
        self.logger.info("Initialized Analysis Layer")
    
    def compute_global_metrics(self) -> Dict[str, Any]:
        """Compute global metrics across all sequences"""
        self.logger.info("Computing global metrics")
        # Get statistics from ClickHouse
        stats = self.clickhouse.get_statistics()
        self.logger.debug(f"Retrieved statistics from ClickHouse: {stats}")
        
        # Query all sequences for detailed analysis
        self.logger.debug("Querying all sequences from ClickHouse")
        all_sequences = self.clickhouse.query_sequences(limit=100000)
        self.logger.info(f"Retrieved {len(all_sequences)} sequences for analysis")
        
        # Separate by source
        openx_sequences = [s for s in all_sequences if s["source"] == "openx"]
        roworks_sequences = [s for s in all_sequences if s["source"] == "roworks"]
        
        # Compute metrics
        metrics = {
            "total_sequences": len(all_sequences),
            "openx_sequences": len(openx_sequences),
            "roworks_sequences": len(roworks_sequences),
            "sequences_by_source": {
                "openx": len(openx_sequences),
                "roworks": len(roworks_sequences)
            },
            "steps_distribution": self._compute_steps_distribution(all_sequences),
            "unique_environments": self._count_unique(all_sequences, "environment_key"),
            "unique_cells": self._count_unique(all_sequences, "cell_key"),
            "unique_activities": self._count_unique(all_sequences, "activity_key"),
            "unique_robots": self._count_unique(all_sequences, "robot_model"),
            "environments_list": list(set(s["environment_key"] for s in all_sequences)),
            "cells_list": list(set(s["cell_key"] for s in all_sequences)),
            "activities_list": list(set(s["activity_key"] for s in all_sequences)),
            "robots_list": list(set(s["robot_model"] for s in all_sequences))
        }
        
        return metrics
    
    def compute_performance_indicators(self) -> Dict[str, Any]:
        """Compute performance indicators"""
        all_sequences = self.clickhouse.query_sequences(limit=100000)
        
        # Filter sequences usable for training
        usable_sequences = [
            s for s in all_sequences
            if settings.MIN_SEQUENCE_LENGTH <= s["num_steps"] <= settings.MAX_SEQUENCE_LENGTH
        ]
        
        # Compute activity complexity
        complexity_metrics = self._compute_activity_complexity(all_sequences)
        
        # Compute spatial density (from LiDAR data)
        spatial_density = self._compute_spatial_density(all_sequences)
        
        indicators = {
            "usable_for_training": {
                "count": len(usable_sequences),
                "percentage": (len(usable_sequences) / len(all_sequences) * 100) if all_sequences else 0,
                "total_sequences": len(all_sequences)
            },
            "avg_frame_resolution": self._compute_avg_frame_resolution(all_sequences),
            "activity_complexity": complexity_metrics,
            "spatial_density": spatial_density
        }
        
        return indicators
    
    def compute_data_kpis(self) -> Dict[str, Any]:
        """Compute data KPIs"""
        all_sequences = self.clickhouse.query_sequences(limit=100000)
        
        openx_sequences = [s for s in all_sequences if s["source"] == "openx"]
        roworks_sequences = [s for s in all_sequences if s["source"] == "roworks"]
        
        # Real factory → synthetic sequence conversion rate
        # This is a placeholder - in reality, you'd track which roworks sequences
        # were converted to synthetic/openx format
        conversion_rate = 0.0  # Would be computed from tracking data
        
        # Unique environment templates from RoWorks
        roworks_environments = list(set(s["environment_key"] for s in roworks_sequences))
        
        # OEM-ready robot-model pairs
        robot_model_pairs = list(set(
            (s["robot_model"], s.get("environment_key", ""))
            for s in all_sequences
        ))
        
        kpis = {
            "factory_to_synthetic_conversion_rate": conversion_rate,
            "unique_roworks_environments": len(roworks_environments),
            "roworks_environments_list": roworks_environments,
            "oem_robot_model_pairs": len(robot_model_pairs),
            "robot_model_pairs_list": robot_model_pairs
        }
        
        return kpis
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics, indicators, and KPIs"""
        return {
            "global_metrics": self.compute_global_metrics(),
            "performance_indicators": self.compute_performance_indicators(),
            "data_kpis": self.compute_data_kpis()
        }
    
    def _compute_steps_distribution(self, sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute distribution of steps per sequence"""
        steps = [s["num_steps"] for s in sequences if s.get("num_steps")]
        
        if not steps:
            return {
                "min": 0,
                "max": 0,
                "mean": 0,
                "median": 0,
                "std": 0,
                "percentiles": {}
            }
        
        return {
            "min": min(steps),
            "max": max(steps),
            "mean": statistics.mean(steps),
            "median": statistics.median(steps),
            "std": statistics.stdev(steps) if len(steps) > 1 else 0,
            "percentiles": {
                "p25": self._percentile(steps, 25),
                "p50": self._percentile(steps, 50),
                "p75": self._percentile(steps, 75),
                "p90": self._percentile(steps, 90),
                "p95": self._percentile(steps, 95)
            },
            "histogram": self._create_histogram(steps, bins=20)
        }
    
    def _count_unique(self, sequences: List[Dict[str, Any]], field: str) -> int:
        """Count unique values for a field"""
        values = [s.get(field) for s in sequences if s.get(field)]
        return len(set(values))
    
    def _compute_activity_complexity(self, sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute average activity complexity"""
        complexities = []
        
        for seq in sequences:
            # Parse objects_used from JSON string
            try:
                objects_used = json.loads(seq.get("objects_used", "[]"))
                num_objects = len(objects_used) if isinstance(objects_used, list) else 0
                
                # Estimate transitions from num_steps
                num_transitions = max(0, seq.get("num_steps", 0) - 1)
                
                complexity = {
                    "num_objects": num_objects,
                    "num_transitions": num_transitions,
                    "total_complexity": num_objects + num_transitions
                }
                complexities.append(complexity)
            except:
                pass
        
        if not complexities:
            return {
                "avg_objects": 0,
                "avg_transitions": 0,
                "avg_total_complexity": 0
            }
        
        return {
            "avg_objects": statistics.mean([c["num_objects"] for c in complexities]),
            "avg_transitions": statistics.mean([c["num_transitions"] for c in complexities]),
            "avg_total_complexity": statistics.mean([c["total_complexity"] for c in complexities])
        }
    
    def _compute_spatial_density(self, sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute spatial density from LiDAR data"""
        # This is a placeholder - in reality, you'd analyze actual LiDAR point clouds
        # For now, we'll estimate based on sequence metadata
        
        roworks_sequences = [s for s in sequences if s["source"] == "roworks"]
        
        # Estimate density from number of steps (more steps = more spatial coverage)
        densities = []
        for seq in roworks_sequences:
            # Higher num_steps might indicate more spatial coverage
            density_estimate = seq.get("num_steps", 0) / 100.0  # Normalize
            densities.append(density_estimate)
        
        if not densities:
            return {
                "avg_density": 0,
                "min_density": 0,
                "max_density": 0
            }
        
        return {
            "avg_density": statistics.mean(densities),
            "min_density": min(densities),
            "max_density": max(densities)
        }
    
    def _compute_avg_frame_resolution(self, sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute average frame resolution"""
        # This is a placeholder - in reality, you'd read actual image dimensions
        # For now, we'll return a default estimate
        
        return {
            "avg_width": 1920,  # Placeholder
            "avg_height": 1080,  # Placeholder
            "avg_pixels": 1920 * 1080
        }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Compute percentile"""
        sorted_data = sorted(data)
        index = (percentile / 100.0) * (len(sorted_data) - 1)
        lower = int(index)
        upper = lower + 1
        
        if upper >= len(sorted_data):
            return sorted_data[-1]
        
        weight = index - lower
        return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight
    
    def _create_histogram(self, data: List[float], bins: int = 20) -> Dict[str, int]:
        """Create histogram"""
        if not data:
            return {}
        
        min_val = min(data)
        max_val = max(data)
        bin_width = (max_val - min_val) / bins if max_val > min_val else 1
        
        histogram = {}
        for val in data:
            bin_idx = int((val - min_val) / bin_width) if bin_width > 0 else 0
            bin_idx = min(bin_idx, bins - 1)
            bin_label = f"{min_val + bin_idx * bin_width:.1f}-{min_val + (bin_idx + 1) * bin_width:.1f}"
            histogram[bin_label] = histogram.get(bin_label, 0) + 1
        
        return histogram

