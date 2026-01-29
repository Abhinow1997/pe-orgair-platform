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

def step8_build_pipeline(state: PipelineState) -> PipelineState:
    """Step 8: Validate pipeline and prepare for final output."""
    if not state.chunked_filings:
        raise ValueError("Pipeline incomplete. Ensure all previous steps completed successfully.")

    print("✓ Pipeline built successfully")
    print(f"  - Total filings processed: {len(state.chunked_filings)}")
    print(f"  - Total chunks created: {sum(f['num_chunks'] for f in state.chunked_filings)}")
    print(f"  - Duplicates skipped: {state.summary['skipped_duplicates']}")
    print(f"  - Parsing errors: {state.summary['parsing_errors']}")
    return state


def step9_generate_report(state: PipelineState, output_dir: str = "./pipeline_output") -> Dict[str, Any]:
    """Step 9: Generate final report and export processed data."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report = {
        "pipeline_summary": {
            "attempted_downloads": state.summary["attempted_downloads"],
            "unique_filings_processed": state.summary["unique_filings_processed"],
            "skipped_duplicates": state.summary["skipped_duplicates"],
            "parsing_errors": state.summary["parsing_errors"],
            "total_chunks": sum(f.get("num_chunks", 0) for f in state.chunked_filings)
        },
        "filings": []
    }

    for filing in state.chunked_filings:
        filing_data = {
            "cik": filing["cik"],
            "filing_type": filing["filing_type"],
            "accession_number": filing["accession_number"],
            "content_hash": filing["content_hash"],
            "num_chunks": filing["num_chunks"],
            "num_tables": len(filing.get("parsed_tables", [])),
            "parsed_tables": filing.get("parsed_tables"),
            "chunks": filing["chunks"]
        }
        report["filings"].append(filing_data)

        filing_output = output_path / f"{filing['cik']}_{filing['filing_type']}_{filing['accession_number']}.json"
        with open(filing_output, 'w') as f:
            json.dump(filing_data, f, indent=2)

    report_file = output_path / "pipeline_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    summary_file = output_path / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("SEC EDGAR Pipeline Summary Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Attempted downloads: {report['pipeline_summary']['attempted_downloads']}\n")
        f.write(f"Unique filings processed: {report['pipeline_summary']['unique_filings_processed']}\n")
        f.write(f"Duplicates skipped: {report['pipeline_summary']['skipped_duplicates']}\n")
        f.write(f"Parsing errors: {report['pipeline_summary']['parsing_errors']}\n")
        f.write(f"Total chunks created: {report['pipeline_summary']['total_chunks']}\n\n")
        f.write("="*60 + "\n")
        f.write("Filing Details\n")
        f.write("="*60 + "\n\n")
        for detail in state.summary["details"]:
            f.write(
                f"CIK: {detail['cik']}, Type: {detail['filing_type']}, "
                f"Acc #: {detail['accession_number']}, Status: {detail['status']}\n"
            )

    print(f"\n✓ Report generated in {output_dir}/")
    print(f"  - Summary: {summary_file}")
    print(f"  - Full report: {report_file}")
    print(f"  - Individual filings: {len(state.chunked_filings)} JSON files")

    return report
