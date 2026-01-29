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

class DocumentRegistry:
    """Manages document deduplication using content hashes."""

    def __init__(self, registry_file: str = "document_registry.txt"):
        self.registry_file = Path(registry_file)
        self.processed_hashes = set()
        self._load_registry()

    def _load_registry(self):
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.processed_hashes = {line.strip() for line in f}
            logger.info("Loaded registry", count=len(self.processed_hashes))

    def _save_registry(self):
        with open(self.registry_file, 'w') as f:
            for h in self.processed_hashes:
                f.write(h + '\n')

    def compute_content_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def is_processed(self, content_hash: str) -> bool:
        return content_hash in self.processed_hashes

    def mark_as_processed(self, content_hash: str):
        if content_hash not in self.processed_hashes:
            self.processed_hashes.add(content_hash)
            self._save_registry()


def step6_deduplicate_documents(state: PipelineState, registry_file: str = "document_registry.txt") -> PipelineState:
    """Step 6: Remove duplicate documents based on content hash."""
    if not state.parsed_filings:
        raise ValueError("No parsed filings. Run step5_parse_documents first.")

    state.registry = DocumentRegistry(registry_file)
    state.deduplicated_filings = []
    skipped = 0

    for filing in state.parsed_filings:
        content_hash = state.registry.compute_content_hash(filing["parsed_text"])
        filing["content_hash"] = content_hash

        if state.registry.is_processed(content_hash):
            skipped += 1
            state.summary["skipped_duplicates"] += 1
            state.summary["details"].append({
                "cik": filing["cik"],
                "filing_type": filing["filing_type"],
                "accession_number": filing["accession_number"],
                "status": "duplicate_skipped",
                "content_hash": content_hash
            })
            logger.info("Skipped duplicate", hash=content_hash[:16])
        else:
            state.registry.mark_as_processed(content_hash)
            state.deduplicated_filings.append(filing)

    print(f"✓ Deduplicated: {len(state.deduplicated_filings)} unique documents ({skipped} duplicates removed)")
    return state
