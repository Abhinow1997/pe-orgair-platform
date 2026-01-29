"""
AWS Boto3 S3 Demo - Loading Data to S3
Aligned with SpringBigData PE OrgAIR Platform patterns
"""

import boto3
from botocore.exceptions import ClientError
from pydantic import SecretStr
from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path
import json
import io


# ============================================================
# 1. CONFIGURATION (Pydantic Settings Pattern from Lab 1)
# ============================================================

from pydantic_settings import SettingsConfigDict

class AWSSettings(BaseSettings):
    """AWS configuration using Pydantic - matches course patterns"""
    AWS_ACCESS_KEY_ID: SecretStr
    AWS_SECRET_ACCESS_KEY: SecretStr
    AWS_REGION: str = "us-east-1"
    S3_BUCKET: str
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# ============================================================
# 2. S3 CLIENT SETUP
# ============================================================

def get_s3_client(settings: AWSSettings):
    """Create authenticated S3 client"""
    return boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID.get_secret_value(),
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY.get_secret_value(),
        region_name=settings.AWS_REGION
    )


# ============================================================
# 3. UPLOAD OPERATIONS
# ============================================================

def upload_file(s3_client, bucket: str, file_path: str, s3_key: str) -> bool:
    """
    Upload a local file to S3
    
    Args:
        s3_client: Boto3 S3 client
        bucket: Target S3 bucket name
        file_path: Local path to file
        s3_key: Destination key (path) in S3
    """
    try:
        s3_client.upload_file(file_path, bucket, s3_key)
        print(f"✓ Uploaded {file_path} → s3://{bucket}/{s3_key}")
        return True
    except ClientError as e:
        print(f"✗ Upload failed: {e}")
        return False


def upload_json_data(s3_client, bucket: str, data: dict, s3_key: str) -> bool:
    """
    Upload JSON data directly to S3 (no local file needed)
    
    Useful for: SEC filing metadata, pipeline reports, processed data
    """
    try:
        json_bytes = json.dumps(data, indent=2).encode("utf-8")
        s3_client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=json_bytes,
            ContentType="application/json"
        )
        print(f"✓ Uploaded JSON → s3://{bucket}/{s3_key}")
        return True
    except ClientError as e:
        print(f"✗ Upload failed: {e}")
        return False


def upload_bytes(s3_client, bucket: str, data: bytes, s3_key: str, 
                 content_type: str = "application/octet-stream") -> bool:
    """
    Upload raw bytes to S3
    
    Useful for: PDF documents, binary files, processed SEC filings
    """
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=data,
            ContentType=content_type
        )
        print(f"✓ Uploaded bytes → s3://{bucket}/{s3_key}")
        return True
    except ClientError as e:
        print(f"✗ Upload failed: {e}")
        return False


def upload_dataframe(s3_client, bucket: str, df, s3_key: str, 
                     format: str = "csv") -> bool:
    """
    Upload pandas DataFrame to S3
    
    Args:
        format: "csv" or "parquet"
    """
    try:
        buffer = io.BytesIO()
        if format == "csv":
            df.to_csv(buffer, index=False)
            content_type = "text/csv"
        elif format == "parquet":
            df.to_parquet(buffer, index=False)
            content_type = "application/octet-stream"
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        buffer.seek(0)
        s3_client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=buffer.getvalue(),
            ContentType=content_type
        )
        print(f"✓ Uploaded DataFrame ({format}) → s3://{bucket}/{s3_key}")
        return True
    except ClientError as e:
        print(f"✗ Upload failed: {e}")
        return False


# ============================================================
# 4. UTILITY FUNCTIONS
# ============================================================

def list_bucket_contents(s3_client, bucket: str, prefix: str = "") -> list:
    """List objects in bucket with optional prefix filter"""
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        objects = response.get("Contents", [])
        for obj in objects:
            print(f"  {obj['Key']} ({obj['Size']} bytes)")
        return objects
    except ClientError as e:
        print(f"✗ List failed: {e}")
        return []


def download_file(s3_client, bucket: str, s3_key: str, local_path: str) -> bool:
    """Download a file from S3 to local path"""
    try:
        s3_client.download_file(bucket, s3_key, local_path)
        print(f"✓ Downloaded s3://{bucket}/{s3_key} → {local_path}")
        return True
    except ClientError as e:
        print(f"✗ Download failed: {e}")
        return False


def check_bucket_exists(s3_client, bucket: str) -> bool:
    """Verify bucket exists and is accessible"""
    try:
        s3_client.head_bucket(Bucket=bucket)
        return True
    except ClientError:
        return False


# ============================================================
# 5. EXAMPLE USAGE (PE OrgAIR Context)
# ============================================================

if __name__ == "__main__":
    # Load settings from environment
    settings = AWSSettings()
    
    # Create S3 client
    s3 = get_s3_client(settings)
    
    # Example 1: Upload SEC filing metadata (from Lab 3 pipeline)
    filing_metadata = {
        "cik": "0000320193",
        "company": "Apple Inc",
        "filing_type": "10-K",
        "filed_date": "2024-11-01",
        "chunks": 145,
        "pipeline_version": "1.0"
    }
    upload_json_data(
        s3, 
        settings.S3_BUCKET, 
        filing_metadata, 
        "sec-filings/metadata/apple_10k_2024.json"
    )
    
    # Example 2: Upload a local file 
    # upload_file(
    # s3, 
    # settings.S3_BUCKET, 
    # "services/sec_filings/sec-edgar-filings/0000320193/10-K/0000320193-25-000079/full-submission.html",  # local path
    # "sec-filings/0000320193/10-K/full-submission.html"  # S3 destination
    #     )

    # download_file(
    #     s3,
    #     settings.S3_BUCKET,
    #     "sec-filings/metadata/apple_10k_2024.json",
    #     "downloaded_apple_10k_2024_metadata.json"
    # )

    # Example 3: Upload processed text chunks
    chunks_data = {
        "document_id": "apple_10k_2024",
        "chunks": [
            {"id": 1, "text": "Risk factors include...", "tokens": 512},
            {"id": 2, "text": "Revenue increased by...", "tokens": 489}
        ]
    }
    upload_json_data(
        s3,
        settings.S3_BUCKET,
        chunks_data,
        "sec-filings/chunks/apple_10k_2024_chunks.json"
    )
    
    # List uploaded files
    print("\n📁 Bucket contents:")
    list_bucket_contents(s3, settings.S3_BUCKET, prefix="sec-filings/")