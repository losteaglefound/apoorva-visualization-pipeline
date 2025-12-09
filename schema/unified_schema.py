"""
RoWorks Unified Activity Schema

Defines the common schema for both Open-X and Bridgedata sources.
This schema enables unified analysis and visualization across data sources.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
from enum import Enum


class DataSource(str, Enum):
    """Data source identifier"""
    OPENX = "openx"
    ROWORKS = "roworks"


@dataclass
class ObjectAsset:
    """Object and asset information"""
    object_id: str
    object_type: str
    taxonomy_path: List[str]
    glb_path: Optional[str] = None
    bounding_box: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepFrame:
    """Individual step frame with vision, language, and kinematic tokens"""
    step_id: int
    timestamp: float
    vision_tokens: Optional[List[str]] = None  # Image paths or embeddings
    language_tokens: Optional[List[str]] = None  # Text descriptions
    kinematic_tokens: Optional[Dict[str, Any]] = None  # Robot state, joint angles, etc.
    lidar_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentTemplate:
    """Environment template information"""
    environment_key: str
    environment_type: str
    bounding_boxes: Optional[List[Dict[str, float]]] = None
    layout_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CellTemplate:
    """Cell template information"""
    cell_key: str
    cell_type: str
    safety_constraints: Optional[List[str]] = None
    payload_constraints: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HumanAction:
    """Human action information"""
    action_id: str
    action_type: str
    timestamp: float
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedSequence:
    """
    Unified sequence schema compatible with both Open-X and Bridgedata.
    
    This is the top-level entity that represents a complete activity sequence.
    """
    # Required fields
    sequence_id: str
    source: Literal["openx", "roworks"]
    activity_key: str
    environment_key: str
    cell_key: str
    num_steps: int
    
    # Objects and assets
    objects_used: List[ObjectAsset]
    
    # Robot information
    robot_model: str
    
    # Human actions
    human_actions: List[HumanAction]
    
    # Sequence data
    steps: List[StepFrame]
    
    # Environment and cell templates
    environment: EnvironmentTemplate
    cell: CellTemplate
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Additional fields from metadata
    sequence_length: Optional[int] = None
    payload: Optional[float] = None
    cycle_time: Optional[float] = None
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/JSON serialization"""
        return {
            "sequence_id": self.sequence_id,
            "source": self.source,
            "activity_key": self.activity_key,
            "environment_key": self.environment_key,
            "cell_key": self.cell_key,
            "num_steps": self.num_steps,
            "sequence_length": self.sequence_length,
            "robot_model": self.robot_model,
            "payload": self.payload,
            "cycle_time": self.cycle_time,
            "objects_used": [
                {
                    "object_id": obj.object_id,
                    "object_type": obj.object_type,
                    "taxonomy_path": obj.taxonomy_path,
                    "glb_path": obj.glb_path,
                    "bounding_box": obj.bounding_box,
                    "metadata": obj.metadata
                }
                for obj in self.objects_used
            ],
            "human_actions": [
                {
                    "action_id": act.action_id,
                    "action_type": act.action_type,
                    "timestamp": act.timestamp,
                    "description": act.description,
                    "metadata": act.metadata
                }
                for act in self.human_actions
            ],
            "environment": {
                "environment_key": self.environment.environment_key,
                "environment_type": self.environment.environment_type,
                "bounding_boxes": self.environment.bounding_boxes,
                "layout_path": self.environment.layout_path,
                "metadata": self.environment.metadata
            },
            "cell": {
                "cell_key": self.cell.cell_key,
                "cell_type": self.cell.cell_type,
                "safety_constraints": self.cell.safety_constraints,
                "payload_constraints": self.cell.payload_constraints,
                "metadata": self.cell.metadata
            },
            "steps": [
                {
                    "step_id": step.step_id,
                    "timestamp": step.timestamp,
                    "vision_tokens": step.vision_tokens,
                    "language_tokens": step.language_tokens,
                    "kinematic_tokens": step.kinematic_tokens,
                    "lidar_path": step.lidar_path,
                    "metadata": step.metadata
                }
                for step in self.steps
            ],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedSequence":
        """Create from dictionary"""
        # Parse objects
        objects_used = [
            ObjectAsset(
                object_id=obj["object_id"],
                object_type=obj["object_type"],
                taxonomy_path=obj.get("taxonomy_path", []),
                glb_path=obj.get("glb_path"),
                bounding_box=obj.get("bounding_box"),
                metadata=obj.get("metadata", {})
            )
            for obj in data.get("objects_used", [])
        ]
        
        # Parse human actions
        human_actions = [
            HumanAction(
                action_id=act["action_id"],
                action_type=act["action_type"],
                timestamp=act["timestamp"],
                description=act.get("description"),
                metadata=act.get("metadata", {})
            )
            for act in data.get("human_actions", [])
        ]
        
        # Parse steps
        steps = [
            StepFrame(
                step_id=step["step_id"],
                timestamp=step["timestamp"],
                vision_tokens=step.get("vision_tokens"),
                language_tokens=step.get("language_tokens"),
                kinematic_tokens=step.get("kinematic_tokens"),
                lidar_path=step.get("lidar_path"),
                metadata=step.get("metadata", {})
            )
            for step in data.get("steps", [])
        ]
        
        # Parse environment
        env_data = data.get("environment", {})
        environment = EnvironmentTemplate(
            environment_key=env_data.get("environment_key", ""),
            environment_type=env_data.get("environment_type", ""),
            bounding_boxes=env_data.get("bounding_boxes"),
            layout_path=env_data.get("layout_path"),
            metadata=env_data.get("metadata", {})
        )
        
        # Parse cell
        cell_data = data.get("cell", {})
        cell = CellTemplate(
            cell_key=cell_data.get("cell_key", ""),
            cell_type=cell_data.get("cell_type", ""),
            safety_constraints=cell_data.get("safety_constraints"),
            payload_constraints=cell_data.get("payload_constraints"),
            metadata=cell_data.get("metadata", {})
        )
        
        # Parse timestamps
        created_at = None
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"])
        
        updated_at = None
        if data.get("updated_at"):
            updated_at = datetime.fromisoformat(data["updated_at"])
        
        return cls(
            sequence_id=data["sequence_id"],
            source=data["source"],
            activity_key=data["activity_key"],
            environment_key=data["environment_key"],
            cell_key=data["cell_key"],
            num_steps=data["num_steps"],
            sequence_length=data.get("sequence_length"),
            objects_used=objects_used,
            robot_model=data["robot_model"],
            human_actions=human_actions,
            steps=steps,
            environment=environment,
            cell=cell,
            payload=data.get("payload"),
            cycle_time=data.get("cycle_time"),
            metadata=data.get("metadata", {}),
            created_at=created_at,
            updated_at=updated_at
        )

