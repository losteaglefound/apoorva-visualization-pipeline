"""Pipeline B: RoWorks Bridgedata ETL - Process and normalize BridgeData V2"""

import json
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import Counter

try:
    import numpy as np
except ImportError:
    np = None

from schema.unified_schema import (
    UnifiedSequence, ObjectAsset, StepFrame, EnvironmentTemplate,
    CellTemplate, HumanAction
)
from storage.clickhouse_client import ClickHouseClient
from storage.s3_client import S3Client
from config.settings import settings
from config.logging_config import get_bridgedata_etl_logger


class BridgedataETL:
    """ETL pipeline for BridgeData V2 (RoWorks Bridgedata)"""
    
    def __init__(self):
        self.logger = get_bridgedata_etl_logger()
        self.data_path = Path(settings.BRIDGEDATA_PATH)
        self.clickhouse = ClickHouseClient()
        self.s3 = S3Client()
        self.processed_count = 0
        self.error_count = 0
        self.logger.info(f"Initialized BridgeData V2 ETL pipeline with data path: {self.data_path}")
    
    def find_trajectory_directories(self) -> List[Path]:
        """
        Find all trajectory directories in BridgeData V2 format.
        
        BridgeData V2 structure:
        - Trajectories are organized in directories
        - Each trajectory contains image files (JPEGs) and possibly NumPy files
        - Images are named with timestep indices
        """
        self.logger.debug(f"Searching for trajectory directories in {self.data_path}")
        trajectory_dirs = []
        
        # Look for directories that contain image files (typical BridgeData structure)
        # Common patterns: directories with multiple .jpg/.jpeg files
        for item in self.data_path.iterdir():
            if not item.is_dir():
                continue
            
            # Check if directory contains image files (indicating a trajectory)
            image_files = list(item.glob("*.jpg")) + list(item.glob("*.jpeg")) + list(item.glob("*.png"))
            numpy_files = list(item.glob("*.npy")) if np else []
            
            # If it has images or numpy files, it's likely a trajectory
            if image_files or numpy_files:
                trajectory_dirs.append(item)
                self.logger.debug(f"Found trajectory directory: {item.name} ({len(image_files)} images, {len(numpy_files)} numpy files)")
        
        # Also check nested structure (some BridgeData might be organized by environment/task)
        for subdir in self.data_path.rglob("*"):
            if subdir.is_dir() and subdir not in trajectory_dirs:
                image_files = list(subdir.glob("*.jpg")) + list(subdir.glob("*.jpeg")) + list(subdir.glob("*.png"))
                if len(image_files) >= 5:  # At least 5 images suggests a trajectory
                    trajectory_dirs.append(subdir)
                    self.logger.debug(f"Found nested trajectory directory: {subdir}")
        
        self.logger.info(f"Found {len(trajectory_dirs)} trajectory directories")
        return trajectory_dirs
    
    def parse_trajectory_images(self, traj_dir: Path) -> List[Dict[str, Any]]:
        """
        Parse image files from a trajectory directory.
        
        BridgeData V2 images are typically named with timestep indices.
        Multiple camera views: over-the-shoulder, randomized, depth, wrist
        """
        frames = []
        
        # Find all image files
        image_files = sorted(traj_dir.glob("*.jpg")) + sorted(traj_dir.glob("*.jpeg")) + sorted(traj_dir.glob("*.png"))
        
        # Group by camera view if naming convention indicates it
        # Common patterns: image_0.jpg, image_1.jpg or camera_view_timestep.jpg
        for idx, img_file in enumerate(image_files):
            frame_data = {
                "step_id": idx,
                "timestamp": float(idx) * 0.2,  # 5 Hz = 0.2s per step (BridgeData control frequency)
                "image_path": str(img_file),
                "image_name": img_file.name,
                "camera_view": self._infer_camera_view(img_file.name)
            }
            frames.append(frame_data)
        
        self.logger.debug(f"Parsed {len(frames)} frames from trajectory {traj_dir.name}")
        return frames
    
    def _infer_camera_view(self, filename: str) -> str:
        """Infer camera view from filename"""
        filename_lower = filename.lower()
        if "wrist" in filename_lower:
            return "wrist"
        elif "depth" in filename_lower:
            return "depth"
        elif "random" in filename_lower or "rand" in filename_lower:
            return "randomized"
        elif "shoulder" in filename_lower or "over" in filename_lower:
            return "over-the-shoulder"
        else:
            return "primary"  # Default primary camera view
    
    def parse_trajectory_numpy(self, traj_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Parse NumPy files if present (actions, states, etc.)
        BridgeData V2 may include NumPy files with trajectory data
        """
        if np is None:
            return None
        
        numpy_data = {}
        numpy_files = list(traj_dir.glob("*.npy"))
        
        for npy_file in numpy_files:
            try:
                data = np.load(npy_file, allow_pickle=True)
                key = npy_file.stem  # Filename without extension
                numpy_data[key] = {
                    "shape": data.shape if hasattr(data, 'shape') else None,
                    "dtype": str(data.dtype) if hasattr(data, 'dtype') else None,
                    "file": str(npy_file)
                }
                self.logger.debug(f"Loaded numpy file: {npy_file.name} with shape {data.shape if hasattr(data, 'shape') else 'N/A'}")
            except Exception as e:
                self.logger.warning(f"Failed to load numpy file {npy_file}: {e}")
        
        return numpy_data if numpy_data else None
    
    def parse_language_annotation(self, traj_dir: Path) -> Optional[str]:
        """
        Parse natural language instruction for the trajectory.
        BridgeData V2 includes language annotations for each trajectory.
        """
        # Look for common annotation file names
        annotation_files = [
            traj_dir / "annotation.txt",
            traj_dir / "instruction.txt",
            traj_dir / "language.txt",
            traj_dir / "task.txt",
            traj_dir / "description.txt"
        ]
        
        for ann_file in annotation_files:
            if ann_file.exists():
                try:
                    with open(ann_file, "r") as f:
                        annotation = f.read().strip()
                    self.logger.debug(f"Found language annotation: {annotation[:50]}...")
                    return annotation
                except Exception as e:
                    self.logger.warning(f"Failed to read annotation file {ann_file}: {e}")
        
        # Check for JSON metadata files
        metadata_files = list(traj_dir.glob("*.json"))
        for meta_file in metadata_files:
            try:
                with open(meta_file, "r") as f:
                    metadata = json.load(f)
                    # Common keys for language annotations
                    for key in ["annotation", "instruction", "language", "task", "description", "goal"]:
                        if key in metadata:
                            annotation = str(metadata[key])
                            self.logger.debug(f"Found language annotation in {meta_file.name}: {annotation[:50]}...")
                            return annotation
            except Exception as e:
                self.logger.debug(f"Could not parse metadata file {meta_file}: {e}")
        
        return None
    
    def infer_environment_from_path(self, traj_dir: Path) -> str:
        """
        Infer environment from trajectory directory path.
        BridgeData V2 has 24 environments grouped into 4 categories.
        """
        path_str = str(traj_dir).lower()
        
        # Check for common environment indicators in path
        if "kitchen" in path_str:
            return "toy_kitchen"
        elif "tabletop" in path_str or "table" in path_str:
            return "tabletop"
        elif "sink" in path_str:
            return "toy_sink"
        elif "laundry" in path_str or "washer" in path_str:
            return "toy_laundry"
        else:
            # Create hash from path for unique identification
            path_hash = hashlib.md5(path_str.encode()).hexdigest()[:8]
            return f"env_{path_hash}"
    
    def detect_activity_from_language(self, language_annotation: Optional[str]) -> str:
        """
        Detect activity from natural language instruction.
        BridgeData V2 has 13 skills: pick-and-place, pushing, sweeping, etc.
        """
        if not language_annotation:
            return "general_manipulation"
        
        annotation_lower = language_annotation.lower()
        
        # Map common phrases to activity types
        if any(word in annotation_lower for word in ["pick", "place", "put", "move"]):
            return "pick_and_place"
        elif any(word in annotation_lower for word in ["push", "slide"]):
            return "pushing"
        elif any(word in annotation_lower for word in ["sweep", "wipe"]):
            return "sweeping"
        elif any(word in annotation_lower for word in ["stack", "pile"]):
            return "stacking"
        elif any(word in annotation_lower for word in ["fold", "folded"]):
            return "folding"
        elif any(word in annotation_lower for word in ["open", "close"]):
            return "door_drawer_manipulation"
        else:
            return "general_manipulation"
    
    def create_unified_sequence(self, traj_dir: Path) -> UnifiedSequence:
        """Convert BridgeData V2 trajectory to UnifiedSequence"""
        # Generate sequence ID from directory name
        sequence_id = f"bridgedata_{traj_dir.name}_{self.processed_count}"
        
        # Parse trajectory data
        frames = self.parse_trajectory_images(traj_dir)
        numpy_data = self.parse_trajectory_numpy(traj_dir)
        language_annotation = self.parse_language_annotation(traj_dir)
        
        # Infer metadata
        environment_key = self.infer_environment_from_path(traj_dir)
        activity_key = self.detect_activity_from_language(language_annotation)
        
        # Create cell key from environment and activity
        cell_key = f"cell_{hashlib.md5(f'{environment_key}_{activity_key}'.encode()).hexdigest()[:8]}"
        
        # Create step frames
        steps = []
        for frame in frames:
            step = StepFrame(
                step_id=frame["step_id"],
                timestamp=frame["timestamp"],
                vision_tokens=[frame["image_path"]],  # Image path for this step
                language_tokens=[language_annotation] if language_annotation else None,
                metadata={
                    "camera_view": frame["camera_view"],
                    "image_name": frame["image_name"],
                    "numpy_data": numpy_data.get(f"step_{frame['step_id']}") if numpy_data else None
                }
            )
            steps.append(step)
        
        # Upload images to S3
        for step in steps:
            if step.vision_tokens and len(step.vision_tokens) > 0:
                image_path = step.vision_tokens[0]
                if os.path.exists(image_path):
                    s3_url = self.s3.upload_preview_frame(
                        sequence_id,
                        step.step_id,
                        image_path,
                        source="roworks"
                    )
                    if s3_url:
                        step.metadata["preview_s3_url"] = s3_url
        
        # Create environment template
        environment = EnvironmentTemplate(
            environment_key=environment_key,
            environment_type="bridgedata_v2",
            metadata={
                "trajectory_path": str(traj_dir),
                "num_cameras": len(set(f.get("camera_view", "primary") for f in frames))
            }
        )
        
        # Create cell template
        cell = CellTemplate(
            cell_key=cell_key,
            cell_type="bridgedata_trajectory",
            metadata={
                "environment": environment_key,
                "activity": activity_key
            }
        )
        
        # Create human action from language annotation (teleoperated = human demonstration)
        human_actions = []
        if language_annotation:
            human_action = HumanAction(
                action_id=f"action_{sequence_id}",
                action_type="teleoperation",
                timestamp=0.0,
                description=language_annotation,
                metadata={"source": "bridgedata_v2"}
            )
            human_actions.append(human_action)
        
        # Create unified sequence
        sequence = UnifiedSequence(
            sequence_id=sequence_id,
            source="roworks",
            activity_key=activity_key,
            environment_key=environment_key,
            cell_key=cell_key,
            num_steps=len(steps),
            sequence_length=len(steps),
            objects_used=[],  # BridgeData V2 doesn't explicitly list objects
            robot_model="widowx_250_6dof",  # BridgeData V2 uses WidowX 250 6DOF
            human_actions=human_actions,
            steps=steps,
            environment=environment,
            cell=cell,
            cycle_time=len(steps) * 0.2,  # 5 Hz control frequency
            metadata={
                "trajectory_directory": str(traj_dir),
                "language_annotation": language_annotation,
                "numpy_data_available": numpy_data is not None,
                "num_images": len(frames),
                "control_frequency_hz": 5,
                "average_timesteps": len(steps),
                "data_source": "bridgedata_v2"
            },
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        return sequence
    
    def process(self) -> Dict[str, Any]:
        """Main processing function"""
        self.logger.info("=" * 60)
        self.logger.info("Starting BridgeData V2 ETL pipeline")
        self.logger.info("=" * 60)
        
        # Find all trajectory directories
        trajectory_dirs = self.find_trajectory_directories()
        self.logger.info(f"Found {len(trajectory_dirs)} trajectory directories")
        
        if len(trajectory_dirs) == 0:
            self.logger.warning("No trajectory directories found. BridgeData V2 should contain directories with image files.")
            self.logger.warning(f"Searched in: {self.data_path}")
            self.logger.warning("Expected structure: directories containing .jpg/.jpeg/.png files")
        
        # Process each trajectory
        unified_sequences = []
        for idx, traj_dir in enumerate(trajectory_dirs):
            try:
                self.logger.debug(f"Processing trajectory {idx + 1}/{len(trajectory_dirs)}: {traj_dir.name}")
                
                # Create unified sequence
                sequence = self.create_unified_sequence(traj_dir)
                self.logger.debug(f"Created unified sequence: {sequence.sequence_id}")
                
                # Upload trajectory data to S3 (if there's a metadata file)
                metadata_files = list(traj_dir.glob("*.json"))
                if metadata_files:
                    self.s3.upload_sequence_data(
                        sequence.sequence_id,
                        str(metadata_files[0]),
                        source="roworks"
                    )
                
                # Store in ClickHouse
                if self.clickhouse.insert_sequence(sequence):
                    unified_sequences.append(sequence)
                    self.processed_count += 1
                    self.logger.debug(f"Successfully stored sequence {sequence.sequence_id}")
                else:
                    self.error_count += 1
                    self.logger.warning(f"Failed to store sequence {sequence.sequence_id}")
                
                if self.processed_count % 10 == 0:
                    self.logger.info(f"Progress: {self.processed_count} trajectories processed...")
            
            except Exception as e:
                self.logger.error(f"Error processing trajectory in {traj_dir}: {e}", exc_info=True)
                self.error_count += 1
        
        self.logger.info("=" * 60)
        self.logger.info(f"BridgeData V2 ETL complete: {self.processed_count} processed, {self.error_count} errors")
        self.logger.info("=" * 60)
        
        result = {
            "processed": self.processed_count,
            "errors": self.error_count,
            "total": len(trajectory_dirs)
        }
        self.logger.info(f"ETL Results: {result}")
        
        return result
