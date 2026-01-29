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

from pe_orgair.models.pipelinestate import PipelineState
from pe_orgair.services.downloader import step1_initialize_pipeline,step2_add_downloader, step3_configure_rate_limiting, step4_download_filings
from pe_orgair.services.parser import step5_parse_documents
from pe_orgair.services.registry import step6_deduplicate_documents
from pe_orgair.services.chucker import step7_chunk_text
from pe_orgair.services.reportgeneration import step9_generate_report, step8_build_pipeline

async def main():
    """Example of running the pipeline step by step."""

    # Step 1: Initialize
    state = step1_initialize_pipeline(
        company_name="QuantInsight Analytics",
        email_address="your_email@quantinsight.com",
        download_dir="./sec_filings"
    )

    # Step 2: Add downloader
    state = step2_add_downloader(state)

    # Step 3: Configure rate limiting
    state = step3_configure_rate_limiting(state, request_delay=0.1)

    # Step 4: Download filings
    state = await step4_download_filings(
        state,
        ciks=["0000320193"],  # Apple
        filing_types=["10-K"],
        after_date="2023-01-01",
        limit=1
    )

    # Step 5: Parse documents
    state = await step5_parse_documents(state)

    # Step 6: Deduplicate
    state = step6_deduplicate_documents(state) 

    # Step 7: Chunk text
    state = await step7_chunk_text(state, chunk_size=750, chunk_overlap=50)

    # Step 8: Build pipeline
    state = step8_build_pipeline(state)

    # Step 9: Generate report
    report = step9_generate_report(state, output_dir="./pipeline_output")

    return state, report

if __name__ == "__main__":
    asyncio.run(main())