"""Pipeline A: Open-X ETL - Normalize and process Open-X data"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from schema.unified_schema import (
    UnifiedSequence, ObjectAsset, StepFrame, EnvironmentTemplate,
    CellTemplate, HumanAction
)
from storage.clickhouse_client import ClickHouseClient
from storage.s3_client import S3Client
from config.settings import settings
from config.logging_config import get_openx_etl_logger


class OpenXETL:
    """ETL pipeline for Open-X data"""
    
    def __init__(self):
        self.logger = get_openx_etl_logger()
        self.data_path = Path(settings.OPENX_DATA_PATH)
        self.clickhouse = ClickHouseClient()
        self.s3 = S3Client()
        self.processed_count = 0
        self.error_count = 0
        self.logger.info(f"Initialized Open-X ETL pipeline with data path: {self.data_path}")
    
    def find_json_index(self) -> Optional[Path]:
        """Find the Open-X JSON index file"""
        self.logger.debug(f"Searching for JSON index in {self.data_path}")
        # Common locations for index files
        possible_names = ["index.json", "sequences.json", "metadata.json", "manifest.json"]
        
        for name in possible_names:
            index_path = self.data_path / name
            if index_path.exists():
                self.logger.info(f"Found JSON index: {index_path}")
                return index_path
        
        # Search recursively
        self.logger.debug("Index not found in root, searching recursively...")
        for json_file in self.data_path.rglob("*.json"):
            if json_file.name.lower() in [n.lower() for n in possible_names]:
                self.logger.info(f"Found JSON index: {json_file}")
                return json_file
        
        self.logger.warning("Could not find JSON index file")
        return None
    
    def read_json_index(self) -> List[Dict[str, Any]]:
        """Read the Open-X JSON index"""
        index_path = self.find_json_index()
        if not index_path:
            error_msg = f"Could not find Open-X JSON index in {self.data_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        self.logger.info(f"Reading JSON index from {index_path}")
        try:
            with open(index_path, "r") as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                self.logger.info(f"Found {len(data)} sequences in index")
                return data
            elif isinstance(data, dict):
                # Try common keys
                for key in ["sequences", "data", "items", "entries"]:
                    if key in data:
                        sequences = data[key]
                        self.logger.info(f"Found {len(sequences)} sequences in key '{key}'")
                        return sequences
                # If it's a single sequence, wrap it
                self.logger.info("Single sequence found, wrapping in list")
                return [data]
            else:
                self.logger.warning(f"Unexpected data type in index: {type(data)}")
                return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON index: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error reading JSON index: {e}", exc_info=True)
            raise
    
    def normalize_path(self, path: str, base_dir: Optional[Path] = None) -> Path:
        """Normalize a path relative to the data directory"""
        if base_dir is None:
            base_dir = self.data_path
        
        if os.path.isabs(path):
            return Path(path)
        else:
            return base_dir / path
    
    def extract_metadata(self, sequence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from Open-X sequence data"""
        metadata = {
            "sequence_length": sequence_data.get("sequence_length"),
            "activity_key": sequence_data.get("activity_key", "unknown"),
            "environment": sequence_data.get("environment", {}),
            "objects": sequence_data.get("objects", []),
            "activity": sequence_data.get("activity", {}),
            "original_metadata": sequence_data
        }
        return metadata
    
    def parse_objects(self, objects_data: List[Dict[str, Any]]) -> List[ObjectAsset]:
        """Parse objects from Open-X data"""
        objects = []
        for obj_data in objects_data:
            obj = ObjectAsset(
                object_id=obj_data.get("object_id", obj_data.get("id", "")),
                object_type=obj_data.get("object_type", obj_data.get("type", "")),
                taxonomy_path=obj_data.get("taxonomy_path", obj_data.get("taxonomy", [])),
                bounding_box=obj_data.get("bounding_box"),
                metadata=obj_data.get("metadata", {})
            )
            objects.append(obj)
        return objects
    
    def parse_steps(self, sequence_data: Dict[str, Any], sequence_id: str) -> List[StepFrame]:
        """Parse step frames from Open-X data"""
        steps = []
        
        # Try different possible keys for steps
        steps_data = (
            sequence_data.get("steps") or
            sequence_data.get("frames") or
            sequence_data.get("sequence") or
            []
        )
        
        for idx, step_data in enumerate(steps_data):
            step = StepFrame(
                step_id=idx,
                timestamp=step_data.get("timestamp", float(idx)),
                vision_tokens=step_data.get("vision_tokens") or step_data.get("image_paths"),
                language_tokens=step_data.get("language_tokens") or step_data.get("descriptions"),
                kinematic_tokens=step_data.get("kinematic_tokens") or step_data.get("robot_state"),
                metadata=step_data.get("metadata", {})
            )
            steps.append(step)
        
        return steps
    
    def create_unified_sequence(self, sequence_data: Dict[str, Any]) -> UnifiedSequence:
        """Convert Open-X sequence data to UnifiedSequence"""
        sequence_id = sequence_data.get("sequence_id") or sequence_data.get("id", f"openx_{self.processed_count}")
        
        # Extract metadata
        metadata = self.extract_metadata(sequence_data)
        
        # Parse components
        objects_used = self.parse_objects(metadata.get("objects", []))
        steps = self.parse_steps(sequence_data, sequence_id)
        
        # Extract environment info
        env_data = metadata.get("environment", {})
        environment = EnvironmentTemplate(
            environment_key=env_data.get("environment_key", env_data.get("key", "unknown")),
            environment_type=env_data.get("type", "unknown"),
            layout_path=env_data.get("layout_path"),
            metadata=env_data
        )
        
        # Extract cell info (may not be present in Open-X)
        cell_key = sequence_data.get("cell_key", "default")
        cell = CellTemplate(
            cell_key=cell_key,
            cell_type=sequence_data.get("cell_type", "unknown"),
            metadata=sequence_data.get("cell_metadata", {})
        )
        
        # Parse human actions
        human_actions = []
        actions_data = sequence_data.get("human_actions", [])
        for act_data in actions_data:
            action = HumanAction(
                action_id=act_data.get("action_id", ""),
                action_type=act_data.get("action_type", ""),
                timestamp=act_data.get("timestamp", 0.0),
                description=act_data.get("description"),
                metadata=act_data.get("metadata", {})
            )
            human_actions.append(action)
        
        # Create unified sequence
        sequence = UnifiedSequence(
            sequence_id=sequence_id,
            source="openx",
            activity_key=metadata.get("activity_key", "unknown"),
            environment_key=environment.environment_key,
            cell_key=cell_key,
            num_steps=len(steps),
            sequence_length=metadata.get("sequence_length", len(steps)),
            objects_used=objects_used,
            robot_model=sequence_data.get("robot_model", "unknown"),
            human_actions=human_actions,
            steps=steps,
            environment=environment,
            cell=cell,
            payload=sequence_data.get("payload"),
            cycle_time=sequence_data.get("cycle_time"),
            metadata=metadata,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        return sequence
    
    def upload_preview_frames(self, sequence: UnifiedSequence) -> None:
        """Upload preview frames to S3"""
        # Upload first, middle, and last frames as previews
        if not sequence.steps:
            return
        
        preview_indices = [0]
        if len(sequence.steps) > 1:
            preview_indices.append(len(sequence.steps) // 2)
        if len(sequence.steps) > 2:
            preview_indices.append(len(sequence.steps) - 1)
        
        for idx in preview_indices:
            step = sequence.steps[idx]
            if step.vision_tokens and len(step.vision_tokens) > 0:
                frame_path = step.vision_tokens[0]
                # Normalize path
                normalized_path = self.normalize_path(frame_path)
                if normalized_path.exists():
                    s3_url = self.s3.upload_preview_frame(
                        sequence.sequence_id,
                        step.step_id,
                        str(normalized_path),
                        source="openx"
                    )
                    if s3_url:
                        # Update step metadata with S3 URL
                        step.metadata["preview_s3_url"] = s3_url
    
    def process(self) -> Dict[str, Any]:
        """Main processing function"""
        self.logger.info("=" * 60)
        self.logger.info("Starting Open-X ETL pipeline")
        self.logger.info("=" * 60)
        
        # Read index
        sequences_data = self.read_json_index()
        self.logger.info(f"Found {len(sequences_data)} sequences in index")
        
        # Process each sequence
        unified_sequences = []
        for idx, seq_data in enumerate(sequences_data):
            try:
                self.logger.debug(f"Processing sequence {idx + 1}/{len(sequences_data)}")
                
                # Create unified sequence
                sequence = self.create_unified_sequence(seq_data)
                self.logger.debug(f"Created unified sequence: {sequence.sequence_id}")
                
                # Upload preview frames
                self.upload_preview_frames(sequence)
                
                # Store in ClickHouse
                if self.clickhouse.insert_sequence(sequence):
                    unified_sequences.append(sequence)
                    self.processed_count += 1
                    self.logger.debug(f"Successfully stored sequence {sequence.sequence_id}")
                else:
                    self.error_count += 1
                    self.logger.warning(f"Failed to store sequence {sequence.sequence_id}")
                
                if self.processed_count % 100 == 0:
                    self.logger.info(f"Progress: {self.processed_count} sequences processed...")
            
            except Exception as e:
                self.logger.error(f"Error processing sequence {idx + 1}: {e}", exc_info=True)
                self.error_count += 1
        
        self.logger.info("=" * 60)
        self.logger.info(f"Open-X ETL complete: {self.processed_count} processed, {self.error_count} errors")
        self.logger.info("=" * 60)
        
        result = {
            "processed": self.processed_count,
            "errors": self.error_count,
            "total": len(sequences_data)
        }
        self.logger.info(f"ETL Results: {result}")
        
        return result

