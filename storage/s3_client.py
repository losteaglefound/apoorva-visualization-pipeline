"""S3 client for storing sequences, frames, and GLB assets"""

import os
from typing import Optional, List
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None

from config.settings import settings
from config.logging_config import get_s3_logger


class S3Client:
    """Client for S3 storage operations"""
    
    def __init__(self):
        self.logger = get_s3_logger()
        
        if boto3 is None:
            error_msg = "boto3 not installed. Install with: pip install boto3"
            self.logger.error(error_msg)
            raise ImportError(error_msg)
        
        self.logger.info(f"Initializing S3 client for bucket: {settings.S3_BUCKET}")
        s3_config = {
            "region_name": settings.S3_REGION
        }
        
        if settings.S3_ENDPOINT:
            s3_config["endpoint_url"] = settings.S3_ENDPOINT
            self.logger.debug(f"Using S3 endpoint: {settings.S3_ENDPOINT}")
        
        if settings.S3_ACCESS_KEY and settings.S3_SECRET_KEY:
            s3_config["aws_access_key_id"] = settings.S3_ACCESS_KEY
            s3_config["aws_secret_access_key"] = settings.S3_SECRET_KEY
            self.logger.debug("Using provided S3 credentials")
        
        try:
            self.s3 = boto3.client("s3", **s3_config)
            self.bucket = settings.S3_BUCKET
            self._ensure_bucket()
            self.logger.info("S3 client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize S3 client: {e}", exc_info=True)
            raise
    
    def _ensure_bucket(self):
        """Ensure bucket exists"""
        try:
            self.s3.head_bucket(Bucket=self.bucket)
            self.logger.debug(f"Bucket {self.bucket} exists")
        except ClientError:
            # Bucket doesn't exist, try to create it
            self.logger.info(f"Bucket {self.bucket} does not exist, attempting to create...")
            try:
                if settings.S3_REGION == "us-east-1":
                    self.s3.create_bucket(Bucket=self.bucket)
                else:
                    self.s3.create_bucket(
                        Bucket=self.bucket,
                        CreateBucketConfiguration={"LocationConstraint": settings.S3_REGION}
                    )
                self.logger.info(f"Successfully created bucket {self.bucket}")
            except Exception as e:
                self.logger.warning(f"Could not create bucket {self.bucket}: {e}")
    
    def upload_file(self, local_path: str, s3_key: str) -> Optional[str]:
        """Upload a file to S3 and return the S3 URL"""
        try:
            self.logger.debug(f"Uploading {local_path} to s3://{self.bucket}/{s3_key}")
            self.s3.upload_file(local_path, self.bucket, s3_key)
            s3_url = f"s3://{self.bucket}/{s3_key}"
            self.logger.debug(f"Successfully uploaded to {s3_url}")
            return s3_url
        except Exception as e:
            self.logger.error(f"Error uploading {local_path} to {s3_key}: {e}", exc_info=True)
            return None
    
    def upload_preview_frame(
        self,
        sequence_id: str,
        step_id: int,
        frame_path: str,
        source: str = "openx"
    ) -> Optional[str]:
        """Upload a preview frame for a sequence step"""
        extension = Path(frame_path).suffix
        s3_key = f"previews/{source}/{sequence_id}/step_{step_id:06d}{extension}"
        return self.upload_file(frame_path, s3_key)
    
    def upload_glb_asset(
        self,
        object_id: str,
        glb_path: str,
        source: str = "roworks"
    ) -> Optional[str]:
        """Upload a GLB asset"""
        s3_key = f"assets/{source}/glb/{object_id}.glb"
        return self.upload_file(glb_path, s3_key)
    
    def upload_sequence_data(
        self,
        sequence_id: str,
        data_path: str,
        source: str = "openx"
    ) -> Optional[str]:
        """Upload sequence data (JSON, video, etc.)"""
        extension = Path(data_path).suffix
        s3_key = f"sequences/{source}/{sequence_id}/data{extension}"
        return self.upload_file(data_path, s3_key)
    
    def upload_lidar_data(
        self,
        sequence_id: str,
        step_id: int,
        lidar_path: str,
        source: str = "roworks"
    ) -> Optional[str]:
        """Upload LiDAR data"""
        extension = Path(lidar_path).suffix
        s3_key = f"lidar/{source}/{sequence_id}/step_{step_id:06d}{extension}"
        return self.upload_file(lidar_path, s3_key)
    
    def list_objects(self, prefix: str) -> List[str]:
        """List objects with a given prefix"""
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            if "Contents" in response:
                return [obj["Key"] for obj in response["Contents"]]
            return []
        except Exception as e:
            print(f"Error listing objects with prefix {prefix}: {e}")
            return []
    
    def get_url(self, s3_key: str, expires_in: int = 3600) -> Optional[str]:
        """Get a presigned URL for an S3 object"""
        try:
            url = self.s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": s3_key},
                ExpiresIn=expires_in
            )
            return url
        except Exception as e:
            print(f"Error generating URL for {s3_key}: {e}")
            return None

