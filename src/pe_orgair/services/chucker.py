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

class DocumentChunker:
    """Splits documents into manageable chunks."""

    def __init__(self, chunk_size: int = 750, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info("DocumentChunker initialized", chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def chunk_document(self, text: str) -> List[str]:
        if not text:
            return []

        words = text.split()
        chunks = []
        start_idx = 0

        while start_idx < len(words):
            end_idx = min(start_idx + self.chunk_size, len(words))
            chunk = " ".join(words[start_idx:end_idx])
            chunks.append(chunk)

            start_idx += (self.chunk_size - self.chunk_overlap)
            if start_idx >= len(words):
                break

        return chunks


async def step7_chunk_text(state: PipelineState, chunk_size: int = 750, chunk_overlap: int = 50) -> PipelineState:
    """Step 7: Split document text into manageable chunks."""
    if not state.deduplicated_filings:
        raise ValueError("No deduplicated filings. Run step6_deduplicate_documents first.")

    state.chunker = DocumentChunker(chunk_size, chunk_overlap)
    state.chunked_filings = []

    for filing in async_tqdm(state.deduplicated_filings, desc="Chunking documents"):
        try:
            chunks = state.chunker.chunk_document(filing["parsed_text"])
            if not chunks:
                raise ValueError(f"No chunks generated for {filing['path']}")

            filing["chunks"] = chunks
            filing["num_chunks"] = len(chunks)
            state.chunked_filings.append(filing)

            state.summary["unique_filings_processed"] += 1
            state.summary["details"].append({
                "cik": filing["cik"],
                "filing_type": filing["filing_type"],
                "accession_number": filing["accession_number"],
                "status": "success",
                "num_chunks": len(chunks),
                "parsed_tables": filing.get("parsed_tables"),
                "num_tables": len(filing.get("parsed_tables", [])),
                "content_hash": filing["content_hash"]
            })

        except Exception as e:
            logger.error("Chunking failed", filing=filing["path"], error=str(e))
            state.summary["details"].append({
                "cik": filing["cik"],
                "filing_type": filing["filing_type"],
                "accession_number": filing["accession_number"],
                "status": "chunking_failed",
                "error": str(e)
            })

    print(f"✓ Chunked {len(state.chunked_filings)} documents into chunks")
    return state
