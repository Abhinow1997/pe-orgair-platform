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

from pe_orgair.models.pipelinestate import PipelineState

logger = structlog.get_logger()

FILING_TYPES = ["10-K", "10-Q", "8-K", "DEF 14A"]

class DocumentParser:
    """Parses SEC filings to extract text and tables."""

    def __init__(self):
        logger.info("DocumentParser initialized")

    def _parse_html(self, file_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract text and tables from HTML."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f, 'lxml')

        for element in soup(['script', 'style']):
            element.extract()

        text = soup.get_text(separator='\n')
        text = re.sub(r'\n\s*\n', '\n', text).strip()

        tables_data = []
        for table in soup.find_all('table'):
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            rows = []
            for tr in table.find_all('tr'):
                if tr.find('th'):
                    continue
                cells = [td.get_text(strip=True) for td in tr.find_all('td')]
                if cells:
                    rows.append(cells)

            if headers and rows:
                tables_data.append({"headers": headers, "rows": rows})
            elif rows:
                tables_data.append({"rows": rows})

        return text, tables_data

    def _parse_pdf(self, file_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract text and tables from PDF."""
        full_text = []
        tables_data = []

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if page_text:
                    full_text.append(page_text)

                page_tables = page.extract_tables()
                if page_tables:
                    for table in page_tables:
                        if table and table[0]:
                            headers = table[0]
                            rows = table[1:]
                            tables_data.append({"page": page_num + 1, "headers": headers, "rows": rows})
                        elif table:
                            tables_data.append({"page": page_num + 1, "rows": table})

        # Fallback to PyMuPDF if needed
        try:
            doc = fitz.open(file_path)
            pymupdf_text = [page.get_text("text") for page in doc]
            doc.close()

            if not full_text or len("".join(full_text)) < len("".join(pymupdf_text)):
                full_text = pymupdf_text
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")

        final_text = "\n".join(full_text).strip()
        final_text = re.sub(r'\n\s*\n', '\n', final_text)
        return final_text, tables_data

    def parse_filing(self, file_path: str) -> Dict[str, Any]:
        """Parse a filing file based on extension."""
        path = Path(file_path)

        if path.suffix in ['.html', '.txt']:
            print("Parsing HTML file:", file_path)
            text, tables = self._parse_html(path)
        elif path.suffix == '.pdf':
            print("Parsing PDF file:", file_path)
            text, tables = self._parse_pdf(path)
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            text = f"Content for {path.name} (unsupported type)"
            tables = []

        return {"text": text, "tables": tables}


async def step5_parse_documents(state: PipelineState) -> PipelineState:
    """Step 5: Parse downloaded documents to extract text and tables."""
    if not state.downloaded_filings:
        raise ValueError("No downloaded filings. Run step4_download_filings first.")

    state.parser = DocumentParser()
    state.parsed_filings = []

    for filing in async_tqdm(state.downloaded_filings, desc="Parsing documents"):
        try:
            parsed = state.parser.parse_filing(filing["path"])
            if not parsed["text"]:
                raise ValueError(f"No text extracted from {filing['path']}")

            filing["parsed_text"] = parsed["text"]
            filing["parsed_tables"] = parsed["tables"]
            state.parsed_filings.append(filing)

        except Exception as e:
            state.summary["parsing_errors"] += 1
            logger.error("Parsing failed", filing=filing["path"], error=str(e))
            state.summary["details"].append({
                "cik": filing["cik"],
                "filing_type": filing["filing_type"],
                "accession_number": filing["accession_number"],
                "status": "parsing_failed",
                "error": str(e)
            })
            raise e

    print(f"✓ Parsed {len(state.parsed_filings)} documents ({state.summary['parsing_errors']} errors)")
    return state
