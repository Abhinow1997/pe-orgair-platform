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

def step1_initialize_pipeline(company_name: str, email_address: str, download_dir: str = "./sec_filings") -> PipelineState:
    """Step 1: Initialize the pipeline with company information."""
    state = PipelineState(company_name, email_address, download_dir)
    print(f"✓ Pipeline initialized for {company_name}")
    return state


def step2_add_downloader(state: PipelineState) -> PipelineState:
    """Step 2: Initialize the SEC downloader component."""
    state.downloader = Downloader(
        company_name=state.company_name,
        email_address=state.email_address,
        download_folder=str(state.download_dir)
    )
    logger.info("SEC Downloader added to pipeline")
    print("✓ SEC Downloader initialized")
    return state

def step3_configure_rate_limiting(state: PipelineState, request_delay: float = 0.1) -> PipelineState:
    """Step 3: Configure rate limiting for SEC requests."""
    state.request_delay = request_delay
    logger.info("Rate limiting configured", request_delay=request_delay)
    print(f"✓ Rate limiting set to {request_delay}s between requests ({1/request_delay:.0f} req/s)")
    return state

def convert_filing_type(html_file_path: str, pdf_path: str) -> str:
    # Convert HTML to PDF using pdfkit
    with open(html_file_path, 'r') as f:
        html_content = f.read()
    options = {"load-error-handling": "ignore"}
    pdfkit.from_string(html_content, pdf_path, options=options)
    return pdf_path

async def step4_download_filings(
    state: PipelineState,
    ciks: List[str],
    filing_types: Optional[List[str]] = None,
    after_date: Optional[str] = None,
    limit: int = 10
) -> PipelineState:
    """Step 4: Download SEC filings for specified companies."""
    if not state.downloader:
        raise ValueError("Downloader not initialized. Run step2_add_downloader first.")

    filing_types = filing_types or ["10-K", "10-Q", "8-K", "DEF 14A"]
    state.downloaded_filings = []

    for cik in async_tqdm(ciks, desc="Downloading CIKs"):
        state.summary["attempted_downloads"] += len(filing_types) * limit

        for filing_type in filing_types:
            try:
                # Rate limiting
                await asyncio.sleep(state.request_delay)

                # Download
                state.downloader.get(filing_type, cik, after=after_date, limit=limit)

                # Find downloaded files
                company_type_dir = state.download_dir / "sec-edgar-filings" / cik / filing_type
                if company_type_dir.exists():
                    for accession_dir in company_type_dir.iterdir():
                        if not accession_dir.is_dir():
                            continue

                        html_file = next(accession_dir.glob("*.html"), None)
                        pdf_file = next(accession_dir.glob("*.pdf"), None)
                        txt_file = next(accession_dir.glob("*.txt"), None)

                        # If we got a .txt that is actually HTML, normalize it
                        if txt_file:
                            txt_content = open(txt_file, "r").read()
                            if "</html>" in txt_content:
                                txt_content = txt_content[:txt_content.index("</html>")+7]
                                with open(txt_file, "w") as f:
                                    f.write(txt_content)

                                html_file = Path(txt_file).with_suffix(".html")
                                with open(html_file, "w") as f:
                                    f.write(txt_content)

                                state.downloaded_filings.append({
                                    "cik": cik,
                                    "filing_type": filing_type,
                                    "accession_number": accession_dir.name,
                                    "path": str(html_file)
                                })

                                # Best-effort HTML→PDF
                                try:
                                    print("Converting HTML to PDF:", html_file)
                                    pdf_file = str(html_file.with_suffix('.pdf'))
                                    convert_filing_type(Path(html_file).absolute(), pdf_file)
                                except Exception as e:
                                    logger.error("Conversion to PDF failed", filing=str(html_file), error=str(e))

                        # If a PDF exists, prefer it (it tends to parse more consistently)
                        if pdf_file and os.path.exists(pdf_file):
                            state.downloaded_filings.append({
                                "cik": cik,
                                "filing_type": filing_type,
                                "accession_number": accession_dir.name,
                                "path": str(pdf_file)
                            })

                logger.info("Downloaded filings", cik=cik, filing_type=filing_type)

            except Exception as e:
                logger.error("Download failed", cik=cik, filing_type=filing_type, error=str(e))

    print(f"✓ Downloaded {len(state.downloaded_filings)} filings")
    return state