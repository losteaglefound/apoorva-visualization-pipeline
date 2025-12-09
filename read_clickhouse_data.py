#!/usr/bin/env python3
"""Example script to read data from ClickHouse"""

from storage.clickhouse_client import ClickHouseClient
import json

def main():
    # Initialize client
    client = ClickHouseClient()
    
    # Get statistics
    print("=== Statistics ===")
    stats = client.get_statistics()
    for stat in stats:
        print(f"\nSource: {stat['source']}")
        print(f"  Total sequences: {stat['total_sequences']}")
        print(f"  Total steps: {stat['total_steps']}")
        print(f"  Avg steps: {stat['avg_steps']:.2f}")
        print(f"  Unique activities: {stat['unique_activities']}")
        print(f"  Unique environments: {stat['unique_environments']}")
    
    # Query sequences
    print("\n=== Sample Sequences ===")
    sequences = client.query_sequences(
        source="roworks",
        limit=5
    )
    
    for seq in sequences:
        print(f"\nSequence ID: {seq['sequence_id']}")
        print(f"  Activity: {seq['activity_key']}")
        print(f"  Environment: {seq['environment_key']}")
        print(f"  Steps: {seq['num_steps']}")
        print(f"  Robot: {seq['robot_model']}")
        
        # Parse JSON metadata
        if seq['metadata']:
            metadata = json.loads(seq['metadata'])
            if 'language_annotation' in metadata:
                print(f"  Language: {metadata['language_annotation'][:50]}...")

if __name__ == "__main__":
    main()