from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import hashlib
import re
import asyncio
import json
import os

import pdfkit
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
import pdfplumber
import fitz  # PyMuPDF
import structlog
from tqdm.asyncio import tqdm as async_tqdm

logger = structlog.get_logger()

FILING_TYPES = ["10-K", "10-Q", "8-K", "DEF 14A"]

class PipelineState:
    """Holds the state of the pipeline across all steps."""

    def __init__(self, company_name: str, email_address: str, download_dir: str = "./sec_filings"):
        self.company_name = company_name
        self.email_address = email_address
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # Components (initialized in later steps)
        self.downloader = None
        self.registry = None
        self.parser = None
        self.chunker = None

        # Pipeline data
        self.downloaded_filings = []
        self.parsed_filings = []
        self.deduplicated_filings = []
        self.chunked_filings = []

        # Summary
        self.summary = {
            "attempted_downloads": 0,
            "unique_filings_processed": 0,
            "skipped_duplicates": 0,
            "parsing_errors": 0,
            "details": []
        }

        logger.info("Pipeline state initialized", company_name=company_name, email_address=email_address)