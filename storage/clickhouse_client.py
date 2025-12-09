"""ClickHouse client for storing and querying unified sequences"""

from typing import List, Dict, Any, Optional
import json
from datetime import datetime

try:
    from clickhouse_driver import Client
except ImportError:
    Client = None

from config.settings import settings
from config.logging_config import get_clickhouse_logger
from schema.unified_schema import UnifiedSequence


class ClickHouseClient:
    """Client for ClickHouse database operations"""
    
    def __init__(self):
        self.logger = get_clickhouse_logger()
        
        if Client is None:
            error_msg = "clickhouse_driver not installed. Install with: pip install clickhouse-driver"
            self.logger.error(error_msg)
            raise ImportError(error_msg)
        
        self.logger.info(f"Connecting to ClickHouse: {settings.CLICKHOUSE_HOST}:{settings.CLICKHOUSE_PORT}")
        try:
            # First connect without database to check/create it
            self.client = Client(
                host=settings.CLICKHOUSE_HOST,
                port=settings.CLICKHOUSE_PORT,
                database="default",  # Connect to default database first
                user=settings.CLICKHOUSE_USER,
                password=settings.CLICKHOUSE_PASSWORD
            )
            self.logger.info("Connected to ClickHouse default database")
            
            # Ensure the target database exists
            self._ensure_database()
            
            # Reconnect to the target database
            self.client = Client(
                host=settings.CLICKHOUSE_HOST,
                port=settings.CLICKHOUSE_PORT,
                database=settings.CLICKHOUSE_DATABASE,
                user=settings.CLICKHOUSE_USER,
                password=settings.CLICKHOUSE_PASSWORD
            )
            self.logger.info(f"Connected to database: {settings.CLICKHOUSE_DATABASE}")
            
            # Now ensure tables exist
            self._ensure_schema()
        except Exception as e:
            self.logger.error(f"Failed to connect to ClickHouse: {e}", exc_info=True)
            raise
    
    def _ensure_database(self):
        """Create database if it doesn't exist"""
        self.logger.debug(f"Ensuring database '{settings.CLICKHOUSE_DATABASE}' exists")
        try:
            create_db_query = f"CREATE DATABASE IF NOT EXISTS {settings.CLICKHOUSE_DATABASE}"
            self.client.execute(create_db_query)
            self.logger.info(f"Database '{settings.CLICKHOUSE_DATABASE}' exists or was created")
        except Exception as e:
            self.logger.error(f"Error creating database '{settings.CLICKHOUSE_DATABASE}': {e}", exc_info=True)
            raise
    
    def _ensure_schema(self):
        """Create tables if they don't exist"""
        self.logger.debug("Ensuring database schema exists")
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS unified_sequences (
            sequence_id String,
            source String,
            activity_key String,
            environment_key String,
            cell_key String,
            num_steps UInt32,
            sequence_length Nullable(UInt32),
            robot_model String,
            payload Nullable(Float64),
            cycle_time Nullable(Float64),
            objects_used String,  -- JSON array
            human_actions String,  -- JSON array
            environment String,    -- JSON object
            cell String,           -- JSON object
            steps_count UInt32,
            metadata String,       -- JSON object
            created_at DateTime,
            updated_at DateTime
        ) ENGINE = MergeTree()
        ORDER BY (source, activity_key, environment_key)
        PARTITION BY source
        """
        
        try:
            self.client.execute(create_table_query)
            self.logger.debug("Created/verified unified_sequences table")
        except Exception as e:
            self.logger.error(f"Error creating unified_sequences table: {e}", exc_info=True)
            raise
        
        # Create index table for faster lookups
        create_index_query = """
        CREATE TABLE IF NOT EXISTS sequence_index (
            sequence_id String,
            source String,
            activity_key String,
            environment_key String,
            cell_key String,
            robot_model String,
            num_steps UInt32,
            created_at DateTime
        ) ENGINE = MergeTree()
        ORDER BY sequence_id
        """
        
        try:
            self.client.execute(create_index_query)
            self.logger.debug("Created/verified sequence_index table")
        except Exception as e:
            self.logger.error(f"Error creating sequence_index table: {e}", exc_info=True)
            raise
    
    def insert_sequence(self, sequence: UnifiedSequence) -> bool:
        """Insert a unified sequence into ClickHouse"""
        try:
            self.logger.debug(f"Inserting sequence {sequence.sequence_id} into ClickHouse")
            data = sequence.to_dict()
            
            insert_query = """
            INSERT INTO unified_sequences VALUES
            """
            
            # Convert ISO datetime strings back to datetime objects for ClickHouse
            created_at = data.get("created_at")
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            elif created_at is None:
                created_at = datetime.now()
            
            updated_at = data.get("updated_at")
            if isinstance(updated_at, str):
                updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
            elif updated_at is None:
                updated_at = datetime.now()
            
            values = (
                data["sequence_id"],
                data["source"],
                data["activity_key"],
                data["environment_key"],
                data["cell_key"],
                data["num_steps"],
                data.get("sequence_length"),
                data["robot_model"],
                data.get("payload"),
                data.get("cycle_time"),
                json.dumps(data["objects_used"]),
                json.dumps(data["human_actions"]),
                json.dumps(data["environment"]),
                json.dumps(data["cell"]),
                len(data["steps"]),
                json.dumps(data["metadata"]),
                created_at,
                updated_at
            )
            
            self.client.execute(insert_query, [values])
            self.logger.debug(f"Inserted sequence {sequence.sequence_id} into unified_sequences")
            
            # Also insert into index
            index_query = """
            INSERT INTO sequence_index VALUES
            """
            index_values = (
                data["sequence_id"],
                data["source"],
                data["activity_key"],
                data["environment_key"],
                data["cell_key"],
                data["robot_model"],
                data["num_steps"],
                created_at  # Use the same datetime object for consistency
            )
            self.client.execute(index_query, [index_values])
            self.logger.debug(f"Inserted sequence {sequence.sequence_id} into sequence_index")
            
            return True
        except Exception as e:
            self.logger.error(f"Error inserting sequence {sequence.sequence_id}: {e}", exc_info=True)
            return False
    
    def batch_insert_sequences(self, sequences: List[UnifiedSequence]) -> int:
        """Insert multiple sequences in batch"""
        success_count = 0
        for sequence in sequences:
            if self.insert_sequence(sequence):
                success_count += 1
        return success_count
    
    def query_sequences(
        self,
        source: Optional[str] = None,
        activity_key: Optional[str] = None,
        environment_key: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Query sequences with filters"""
        conditions = []
        if source:
            conditions.append(f"source = '{source}'")
        if activity_key:
            conditions.append(f"activity_key = '{activity_key}'")
        if environment_key:
            conditions.append(f"environment_key = '{environment_key}'")
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
        SELECT * FROM unified_sequences
        WHERE {where_clause}
        LIMIT {limit}
        """
        
        result = self.client.execute(query)
        columns = [
            "sequence_id", "source", "activity_key", "environment_key", "cell_key",
            "num_steps", "sequence_length", "robot_model", "payload", "cycle_time",
            "objects_used", "human_actions", "environment", "cell", "steps_count",
            "metadata", "created_at", "updated_at"
        ]
        
        return [
            dict(zip(columns, row))
            for row in result
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get basic statistics from the database"""
        stats_query = """
        SELECT
            source,
            count() as total_sequences,
            sum(num_steps) as total_steps,
            avg(num_steps) as avg_steps,
            count(DISTINCT activity_key) as unique_activities,
            count(DISTINCT environment_key) as unique_environments,
            count(DISTINCT cell_key) as unique_cells,
            count(DISTINCT robot_model) as unique_robots
        FROM unified_sequences
        GROUP BY source
        """
        
        result = self.client.execute(stats_query)
        columns = [
            "source", "total_sequences", "total_steps", "avg_steps",
            "unique_activities", "unique_environments", "unique_cells", "unique_robots"
        ]
        
        return [
            dict(zip(columns, row))
            for row in result
        ]

