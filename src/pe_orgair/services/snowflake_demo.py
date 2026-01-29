"""
Snowflake Demo Service - PE Org-AI-R Platform
=============================================
Demonstrates: Warehouse, Database, Schema, Table creation and data insertion.

Location: src/pe_orgair/services/snowflake_demo.py
"""

import snowflake.connector
from snowflake.connector import SnowflakeConnection
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

# ============================================================================
# CONFIGURATION - Loaded from .env via Pydantic Settings
# ============================================================================
from pe_orgair.config.settings import get_settings

# Load settings from .env file
settings = get_settings()

# Build Snowflake config from settings
SNOWFLAKE_CONFIG = {
    "account": settings.SNOWFLAKE_ACCOUNT,
    "user": settings.SNOWFLAKE_USER,
    "password": settings.SNOWFLAKE_PASSWORD.get_secret_value(),  # Extract actual password
    "role": settings.SNOWFLAKE_ROLE,
}

# Resource names from settings
DEMO_WAREHOUSE = settings.SNOWFLAKE_WAREHOUSE
DEMO_DATABASE = settings.SNOWFLAKE_DATABASE
DEMO_SCHEMA = settings.SNOWFLAKE_SCHEMA


# ============================================================================
# STEP 1: CONNECTION HELPER
# ============================================================================
def get_snowflake_connection(
    database: Optional[str] = None,
    schema: Optional[str] = None,
    warehouse: Optional[str] = None
) -> SnowflakeConnection:
    """Create a Snowflake connection with optional database/schema/warehouse."""
    config = SNOWFLAKE_CONFIG.copy()
    if database:
        config["database"] = database
    if schema:
        config["schema"] = schema
    if warehouse:
        config["warehouse"] = warehouse
    
    return snowflake.connector.connect(**config)


def execute_sql(conn: SnowflakeConnection, sql: str, params: tuple = None, fetch: bool = False) -> Optional[List]:
    """Execute SQL and optionally fetch results. Supports parameterized queries."""
    cursor = conn.cursor()
    try:
        if params:
            cursor.execute(sql, params)
        else:
            cursor.execute(sql)
        if fetch:
            return cursor.fetchall()
        return None
    finally:
        cursor.close()


# ============================================================================
# STEP 2: CREATE WAREHOUSE
# ============================================================================
def create_warehouse(conn: SnowflakeConnection) -> bool:
    """
    Create a virtual warehouse for compute resources.
    
    Warehouse sizes: X-SMALL, SMALL, MEDIUM, LARGE, X-LARGE, etc.
    AUTO_SUSPEND: Seconds of inactivity before suspending (saves credits)
    AUTO_RESUME: Automatically resume when queries arrive
    """
    print("\n" + "="*60)
    print("📦 STEP 2: Creating Warehouse")
    print("="*60)
    
    sql = f"""
    CREATE WAREHOUSE IF NOT EXISTS {DEMO_WAREHOUSE}
        WITH 
        WAREHOUSE_SIZE = 'X-SMALL'
        AUTO_SUSPEND = 60
        AUTO_RESUME = TRUE
        INITIALLY_SUSPENDED = FALSE
        COMMENT = 'PE Org-AI-R Platform - Demo Warehouse'
    """
    
    execute_sql(conn, sql)
    print(f"✅ Warehouse '{DEMO_WAREHOUSE}' created successfully!")
    
    # Use the warehouse
    execute_sql(conn, f"USE WAREHOUSE {DEMO_WAREHOUSE}")
    print(f"✅ Now using warehouse: {DEMO_WAREHOUSE}")
    
    return True


# ============================================================================
# STEP 3: CREATE DATABASE
# ============================================================================
def create_database(conn: SnowflakeConnection) -> bool:
    """
    Create a database to organize schemas and tables.
    
    Databases contain schemas, which contain tables/views.
    """
    print("\n" + "="*60)
    print("🗄️  STEP 3: Creating Database")
    print("="*60)
    
    sql = f"""
    CREATE DATABASE IF NOT EXISTS {DEMO_DATABASE}
        COMMENT = 'PE Org-AI-R Platform - SEC Filings & PE Intelligence'
    """
    
    execute_sql(conn, sql)
    print(f"✅ Database '{DEMO_DATABASE}' created successfully!")
    
    # Use the database
    execute_sql(conn, f"USE DATABASE {DEMO_DATABASE}")
    print(f"✅ Now using database: {DEMO_DATABASE}")
    
    return True


# ============================================================================
# STEP 4: CREATE SCHEMA
# ============================================================================
def create_schema(conn: SnowflakeConnection) -> bool:
    """
    Create a schema to logically group related tables.
    
    Schema = namespace for tables within a database.
    """
    print("\n" + "="*60)
    print("📁 STEP 4: Creating Schema")
    print("="*60)
    
    sql = f"""
    CREATE SCHEMA IF NOT EXISTS {DEMO_DATABASE}.{DEMO_SCHEMA}
        COMMENT = 'SEC EDGAR filings and parsed documents'
    """
    
    execute_sql(conn, sql)
    print(f"✅ Schema '{DEMO_SCHEMA}' created successfully!")
    
    # Use the schema
    execute_sql(conn, f"USE SCHEMA {DEMO_DATABASE}.{DEMO_SCHEMA}")
    print(f"✅ Now using schema: {DEMO_DATABASE}.{DEMO_SCHEMA}")
    
    return True


# ============================================================================
# STEP 5: CREATE TABLES
# ============================================================================
def create_tables(conn: SnowflakeConnection) -> bool:
    """
    Create tables for SEC filings pipeline data.
    
    Tables:
    1. SEC_FILINGS - Raw filing metadata
    2. FILING_CHUNKS - Parsed/chunked document sections
    3. FOCUS_GROUPS - PE sector configurations
    """
    print("\n" + "="*60)
    print("📊 STEP 5: Creating Tables")
    print("="*60)
    
    # Table 1: SEC Filings (metadata)
    sql_filings = f"""
    CREATE TABLE IF NOT EXISTS {DEMO_DATABASE}.{DEMO_SCHEMA}.SEC_FILINGS (
        filing_id VARCHAR(100) PRIMARY KEY,
        cik VARCHAR(20) NOT NULL,
        company_name VARCHAR(255),
        filing_type VARCHAR(20) NOT NULL,
        filing_date DATE,
        accession_number VARCHAR(50),
        file_path VARCHAR(500),
        file_size_bytes INTEGER,
        status VARCHAR(20) DEFAULT 'downloaded',
        created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
        updated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
    )
    COMMENT = 'SEC EDGAR filing metadata and download status'
    """
    execute_sql(conn, sql_filings)
    print("✅ Table 'SEC_FILINGS' created!")
    
    # Table 2: Filing Chunks (parsed sections)
    sql_chunks = f"""
    CREATE TABLE IF NOT EXISTS {DEMO_DATABASE}.{DEMO_SCHEMA}.FILING_CHUNKS (
        chunk_id VARCHAR(100) PRIMARY KEY,
        filing_id VARCHAR(100) REFERENCES SEC_FILINGS(filing_id),
        chunk_index INTEGER NOT NULL,
        section_name VARCHAR(100),
        content_text TEXT,
        token_count INTEGER,
        embedding ARRAY,
        created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
    )
    COMMENT = 'Parsed and chunked sections from SEC filings'
    """
    execute_sql(conn, sql_chunks)
    print("✅ Table 'FILING_CHUNKS' created!")
    
    # Table 3: Focus Groups (PE sectors)
    sql_focus_groups = f"""
    CREATE TABLE IF NOT EXISTS {DEMO_DATABASE}.{DEMO_SCHEMA}.FOCUS_GROUPS (
        focus_group_id VARCHAR(50) PRIMARY KEY,
        group_name VARCHAR(100) NOT NULL,
        group_code VARCHAR(10) NOT NULL UNIQUE,
        description TEXT,
        display_order INTEGER,
        is_active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
    )
    COMMENT = '7 PE sectors for organization classification'
    """
    execute_sql(conn, sql_focus_groups)
    print("✅ Table 'FOCUS_GROUPS' created!")
    
    # Table 4: Organizations
    sql_organizations = f"""
    CREATE TABLE IF NOT EXISTS {DEMO_DATABASE}.{DEMO_SCHEMA}.ORGANIZATIONS (
        organization_id VARCHAR(50) PRIMARY KEY,
        cik VARCHAR(20),
        name VARCHAR(255) NOT NULL,
        ticker VARCHAR(10),
        focus_group_id VARCHAR(50) REFERENCES FOCUS_GROUPS(focus_group_id),
        industry VARCHAR(100),
        founded_year INTEGER,
        headquarters VARCHAR(255),
        employee_count INTEGER,
        annual_revenue_millions DECIMAL(15,2),
        ai_readiness_score DECIMAL(5,2),
        created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
        updated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
    )
    COMMENT = 'Organizations tracked for PE intelligence'
    """
    execute_sql(conn, sql_organizations)
    print("✅ Table 'ORGANIZATIONS' created!")
    
    return True


# ============================================================================
# STEP 6: INSERT SAMPLE DATA
# ============================================================================
def insert_sample_data(conn: SnowflakeConnection) -> bool:
    """
    Insert sample data into all tables.
    """
    print("\n" + "="*60)
    print("📝 STEP 6: Inserting Sample Data")
    print("="*60)
    
    # Insert Focus Groups (7 PE Sectors) using parameterized queries
    focus_groups = [
        ("fg_mfg", "Manufacturing", "MFG", "Industrial and manufacturing companies", 1),
        ("fg_fin", "Financial Services", "FIN", "Banks, insurance, asset management", 2),
        ("fg_hc", "Healthcare", "HC", "Healthcare providers and life sciences", 3),
        ("fg_tech", "Technology", "TECH", "Software, hardware, IT services", 4),
        ("fg_rtl", "Retail & Consumer", "RTL", "Retail, consumer goods, e-commerce", 5),
        ("fg_enr", "Energy & Utilities", "ENR", "Oil, gas, renewables, utilities", 6),
        ("fg_ps", "Professional Services", "PS", "Consulting, legal, accounting", 7),
    ]
    
    for fg in focus_groups:
        sql = f"""
        MERGE INTO {DEMO_DATABASE}.{DEMO_SCHEMA}.FOCUS_GROUPS AS target
        USING (SELECT %s AS focus_group_id, %s AS group_name, %s AS group_code, 
                      %s AS description, %s AS display_order) AS source
        ON target.focus_group_id = source.focus_group_id
        WHEN NOT MATCHED THEN
            INSERT (focus_group_id, group_name, group_code, description, display_order)
            VALUES (source.focus_group_id, source.group_name, source.group_code, 
                    source.description, source.display_order)
        """
        execute_sql(conn, sql, params=(fg[0], fg[1], fg[2], fg[3], fg[4]))
    print(f"✅ Inserted {len(focus_groups)} focus groups (PE sectors)")
    
    # Insert Sample SEC Filings using parameterized queries
    filings = [
        ("AAPL_10K_2025", "0000320193", "Apple Inc.", "10-K", "2025-01-15", "0000320193-25-000079"),
        ("MSFT_10K_2025", "0000789019", "Microsoft Corporation", "10-K", "2025-01-20", "0000789019-25-000012"),
        ("GOOGL_10K_2025", "0001652044", "Alphabet Inc.", "10-K", "2025-01-18", "0001652044-25-000008"),
    ]
    
    for f in filings:
        sql = f"""
        MERGE INTO {DEMO_DATABASE}.{DEMO_SCHEMA}.SEC_FILINGS AS target
        USING (SELECT %s AS filing_id, %s AS cik, %s AS company_name, 
                      %s AS filing_type, %s AS filing_date, %s AS accession_number) AS source
        ON target.filing_id = source.filing_id
        WHEN NOT MATCHED THEN
            INSERT (filing_id, cik, company_name, filing_type, filing_date, accession_number, status)
            VALUES (source.filing_id, source.cik, source.company_name, 
                    source.filing_type, source.filing_date, source.accession_number, 'processed')
        """
        execute_sql(conn, sql, params=(f[0], f[1], f[2], f[3], f[4], f[5]))
    print(f"✅ Inserted {len(filings)} SEC filings")
    
    # Insert Sample Organizations using parameterized queries
    organizations = [
        ("org_apple", "0000320193", "Apple Inc.", "AAPL", "fg_tech", "Consumer Electronics", 1976, "Cupertino, CA", 164000, 383285.00, 92.5),
        ("org_msft", "0000789019", "Microsoft Corporation", "MSFT", "fg_tech", "Software", 1975, "Redmond, WA", 221000, 211915.00, 95.0),
        ("org_jpm", "0000019617", "JPMorgan Chase & Co.", "JPM", "fg_fin", "Banking", 1799, "New York, NY", 293723, 154792.00, 88.0),
    ]
    
    for o in organizations:
        sql = f"""
        MERGE INTO {DEMO_DATABASE}.{DEMO_SCHEMA}.ORGANIZATIONS AS target
        USING (SELECT %s AS organization_id, %s AS cik, %s AS name, %s AS ticker,
                      %s AS focus_group_id, %s AS industry, %s AS founded_year,
                      %s AS headquarters, %s AS employee_count, 
                      %s AS annual_revenue_millions, %s AS ai_readiness_score) AS source
        ON target.organization_id = source.organization_id
        WHEN NOT MATCHED THEN
            INSERT (organization_id, cik, name, ticker, focus_group_id, industry, 
                    founded_year, headquarters, employee_count, annual_revenue_millions, ai_readiness_score)
            VALUES (source.organization_id, source.cik, source.name, source.ticker,
                    source.focus_group_id, source.industry, source.founded_year,
                    source.headquarters, source.employee_count, 
                    source.annual_revenue_millions, source.ai_readiness_score)
        """
        execute_sql(conn, sql, params=(o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8], o[9], o[10]))
    print(f"✅ Inserted {len(organizations)} organizations")
    
    # Insert Sample Filing Chunks (using parameterized queries for text with special chars)
    chunks = [
        ("chunk_aapl_001", "AAPL_10K_2025", 1, "Business Overview", "Apple Inc. designs, manufactures, and markets smartphones, personal computers...", 150),
        ("chunk_aapl_002", "AAPL_10K_2025", 2, "Risk Factors", "The Companys business, reputation, results of operations, financial condition...", 200),
        ("chunk_msft_001", "MSFT_10K_2025", 1, "Business Overview", "Microsoft Corporation is a technology company. The Company develops and supports...", 175),
    ]
    
    for c in chunks:
        # Use parameterized query with MERGE to handle duplicates
        sql = f"""
        MERGE INTO {DEMO_DATABASE}.{DEMO_SCHEMA}.FILING_CHUNKS AS target
        USING (SELECT %s AS chunk_id, %s AS filing_id, %s AS chunk_index, 
                      %s AS section_name, %s AS content_text, %s AS token_count) AS source
        ON target.chunk_id = source.chunk_id
        WHEN NOT MATCHED THEN
            INSERT (chunk_id, filing_id, chunk_index, section_name, content_text, token_count)
            VALUES (source.chunk_id, source.filing_id, source.chunk_index, 
                    source.section_name, source.content_text, source.token_count)
        """
        execute_sql(conn, sql, params=(c[0], c[1], c[2], c[3], c[4], c[5]))
    print(f"✅ Inserted {len(chunks)} filing chunks")
    
    return True


# ============================================================================
# STEP 7: VERIFY DATA
# ============================================================================
def verify_data(conn: SnowflakeConnection) -> Dict[str, Any]:
    """Query and display the inserted data."""
    print("\n" + "="*60)
    print("🔍 STEP 7: Verifying Data")
    print("="*60)
    
    results = {}
    
    # Count records in each table
    tables = ["FOCUS_GROUPS", "SEC_FILINGS", "ORGANIZATIONS", "FILING_CHUNKS"]
    
    for table in tables:
        sql = f"SELECT COUNT(*) FROM {DEMO_DATABASE}.{DEMO_SCHEMA}.{table}"
        count = execute_sql(conn, sql, fetch=True)[0][0]
        results[table] = count
        print(f"📊 {table}: {count} records")
    
    # Sample query: Organizations with their PE sector
    print("\n📋 Sample Query - Organizations with PE Sectors:")
    sql = f"""
    SELECT o.name, o.ticker, fg.group_name, o.ai_readiness_score
    FROM {DEMO_DATABASE}.{DEMO_SCHEMA}.ORGANIZATIONS o
    JOIN {DEMO_DATABASE}.{DEMO_SCHEMA}.FOCUS_GROUPS fg 
        ON o.focus_group_id = fg.focus_group_id
    ORDER BY o.ai_readiness_score DESC
    """
    rows = execute_sql(conn, sql, fetch=True)
    for row in rows:
        print(f"   • {row[0]} ({row[1]}) - {row[2]} - AI Score: {row[3]}")
    
    return results


# ============================================================================
# STEP 8: CLEANUP (Optional)
# ============================================================================
def cleanup_demo(conn: SnowflakeConnection, drop_all: bool = False) -> bool:
    """
    Clean up demo resources.
    
    WARNING: drop_all=True will delete the warehouse and database!
    """
    print("\n" + "="*60)
    print("🧹 STEP 8: Cleanup (Optional)")
    print("="*60)
    
    if drop_all:
        execute_sql(conn, f"DROP DATABASE IF EXISTS {DEMO_DATABASE}")
        print(f"✅ Database '{DEMO_DATABASE}' dropped")
        
        execute_sql(conn, f"DROP WAREHOUSE IF EXISTS {DEMO_WAREHOUSE}")
        print(f"✅ Warehouse '{DEMO_WAREHOUSE}' dropped")
    else:
        print("ℹ️  Skipping cleanup. Set drop_all=True to remove resources.")
    
    return True


# ============================================================================
# MAIN: RUN THE DEMO
# ============================================================================
def run_demo():
    """Execute the complete Snowflake demo."""
    print("\n" + "="*60)
    print("🚀 SNOWFLAKE DEMO - PE Org-AI-R Platform")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    conn = None
    try:
        # Step 1: Connect
        print("\n" + "="*60)
        print("🔌 STEP 1: Connecting to Snowflake")
        print("="*60)
        conn = get_snowflake_connection()
        print("✅ Connected to Snowflake successfully!")
        
        # Step 2-7: Create resources and insert data
        create_warehouse(conn)
        create_database(conn)
        create_schema(conn)
        create_tables(conn)
        insert_sample_data(conn)
        verify_data(conn)
        
        # Step 8: Cleanup (disabled by default)
        cleanup_demo(conn, drop_all=False)
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
    finally:
        if conn:
            conn.close()
            print("🔌 Connection closed.")


if __name__ == "__main__":
    run_demo()