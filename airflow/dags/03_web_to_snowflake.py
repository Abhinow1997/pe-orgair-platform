"""
JobSpy to S3 to Snowflake DAG
==============================
This DAG demonstrates:
1. Using Airflow Variables for configuration
2. Scraping jobs using JobSpy library
3. Uploading to AWS S3
4. Creating Snowflake stage, table, and loading data
5. Clean separation of config and code

Pipeline Flow:
scrape_jobs → upload_to_s3 → create_stage → create_table → load_to_snowflake → cleanup → log_summary
"""

from airflow import DAG
from airflow.models import Variable
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
from datetime import datetime, timedelta
import csv
import os
import json

# =============================================================================
# DEFAULT ARGUMENTS
# =============================================================================
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
    'email_on_retry': False,
}

# =============================================================================
# CONFIGURATION HELPER
# =============================================================================
def get_config():
    """Fetch all configuration from Airflow Variables with defaults"""
    return {
        # JobSpy settings
        'search_term': Variable.get("jobspy_search_term", default_var="software engineer"),
        'location': Variable.get("jobspy_location", default_var="San Francisco, CA"),
        'results_wanted': int(Variable.get("jobspy_results_wanted", default_var="20")),
        'hours_old': int(Variable.get("jobspy_hours_old", default_var="72")),
        'country': Variable.get("jobspy_country", default_var="USA"),
        'sites': json.loads(Variable.get("jobspy_sites", default_var='["indeed", "linkedin", "google"]')),
        
        # S3 settings
        's3_bucket': Variable.get("jobspy_s3_bucket", default_var="pdfparserdataset"),
        's3_prefix': Variable.get("jobspy_s3_prefix", default_var="jobs/raw"),
        'aws_conn_id': Variable.get("jobspy_aws_conn_id", default_var="aws_default"),
        
        # Snowflake settings
        'snowflake_conn_id': Variable.get("jobspy_snowflake_conn_id", default_var="snowflake_default"),
        'snowflake_database': Variable.get("jobspy_snowflake_database", default_var="PE_ORGAIR_DB"),
        'snowflake_schema': Variable.get("jobspy_snowflake_schema", default_var="SEC_FILINGS"),
        'stage_name': Variable.get("jobspy_stage_name", default_var="jobspy_s3_stage"),
        'table_name': Variable.get("jobspy_table_name", default_var="job_listings"),
    }

# =============================================================================
# DEFINE COLUMNS (JobSpy output columns in correct order)
# =============================================================================
JOBSPY_COLUMNS = [
    'site',
    'job_url',
    'job_url_direct', 
    'title',
    'company',
    'location',
    'job_type',
    'date_posted',
    'salary_source',
    'interval',
    'min_amount',
    'max_amount',
    'currency',
    'is_remote',
    'description',
    'company_url',
    'company_url_direct',
    'company_addresses',
    'company_industry',
    'company_num_employees',
    'company_revenue',
    'company_description',
    'logo_photo_url',
    'banner_photo_url',
    'ceo_name',
    'ceo_photo_url',
    'emails',
    'job_level',
    'job_function',
    'listing_type',
    'company_rating',
    'company_reviews_count',
]

# =============================================================================
# TASK FUNCTIONS
# =============================================================================

def scrape_jobs_task(**context):
    """
    Task 1: Scrape jobs using JobSpy
    - Reads config from Airflow Variables
    - Fetches job listings from multiple sites
    - Saves to local CSV with correct column order
    """
    from jobspy import scrape_jobs
    import pandas as pd
    
    config = get_config()
    exec_date = context['ds']
    
    print(f"=== Configuration from Airflow Variables ===")
    print(f"Search Term:     {config['search_term']}")
    print(f"Location:        {config['location']}")
    print(f"Sites:           {config['sites']}")
    print(f"Results Wanted:  {config['results_wanted']}")
    print(f"Hours Old:       {config['hours_old']}")
    print(f"============================================")
    
    # Scrape jobs
    jobs = scrape_jobs(
        site_name=config['sites'],
        search_term=config['search_term'],
        google_search_term=f"{config['search_term']} jobs near {config['location']} since yesterday",
        location=config['location'],
        results_wanted=config['results_wanted'],
        hours_old=config['hours_old'],
        country_indeed=config['country'],
    )
    
    print(f"Found {len(jobs)} jobs")
    print(f"Columns in scraped data: {jobs.columns.tolist()}")
    
    # Ensure column order matches Snowflake table
    # Add missing columns with None, remove extra columns
    for col in JOBSPY_COLUMNS:
        if col not in jobs.columns:
            jobs[col] = None
    
    # Reorder columns to match Snowflake table
    jobs = jobs[[col for col in JOBSPY_COLUMNS if col in jobs.columns]]
    
    print(f"Final columns (ordered): {jobs.columns.tolist()}")
    
    # Save to local CSV
    local_file = f"/tmp/jobs_{exec_date}.csv"
    jobs.to_csv(
        local_file, 
        quoting=csv.QUOTE_NONNUMERIC, 
        escapechar="\\", 
        index=False
    )
    
    # Push to XCom
    context['task_instance'].xcom_push(key='local_file', value=local_file)
    context['task_instance'].xcom_push(key='job_count', value=len(jobs))
    context['task_instance'].xcom_push(key='columns', value=jobs.columns.tolist())
    
    return f"Scraped {len(jobs)} jobs"


def upload_to_s3_task(**context):
    """
    Task 2: Upload CSV to S3
    - Reads S3 config from Airflow Variables
    - Uploads with date partitioning
    """
    config = get_config()
    exec_date = context['ds']
    
    # Pull from XCom
    local_file = context['task_instance'].xcom_pull(
        task_ids='scrape_jobs',
        key='local_file'
    )
    
    year, month, day = exec_date.split('-')
    
    # Create S3 key with date partitioning
    s3_key = f"{config['s3_prefix']}/year={year}/month={month}/day={day}/jobs_{exec_date}.csv"
    
    print(f"Uploading to s3://{config['s3_bucket']}/{s3_key}")
    
    # Upload using S3Hook
    s3_hook = S3Hook(aws_conn_id=config['aws_conn_id'])
    s3_hook.load_file(
        filename=local_file,
        key=s3_key,
        bucket_name=config['s3_bucket'],
        replace=True
    )
    
    s3_path = f"s3://{config['s3_bucket']}/{s3_key}"
    context['task_instance'].xcom_push(key='s3_path', value=s3_path)
    context['task_instance'].xcom_push(key='s3_key', value=s3_key)
    
    return f"Uploaded to {s3_path}"


def create_snowflake_stage_task(**context):
    """
    Task 3: Create S3 external stage in Snowflake (if not exists)
    - Fetches AWS credentials from Airflow Connection
    - Creates stage pointing to S3 bucket
    """
    config = get_config()
    
    print("Creating Snowflake stage (if not exists)...")
    
    # Get AWS credentials from Airflow Connection
    s3_hook = S3Hook(aws_conn_id=config['aws_conn_id'])
    credentials = s3_hook.get_credentials()
    
    aws_key_id = credentials.access_key
    aws_secret_key = credentials.secret_key
    
    if not aws_key_id or not aws_secret_key:
        raise ValueError("AWS credentials not found in connection")
    
    # Build the stage URL
    stage_url = f"s3://{config['s3_bucket']}/{config['s3_prefix']}/"
    
    # Create stage SQL (CREATE IF NOT EXISTS equivalent)
    create_stage_sql = f"""
    CREATE STAGE IF NOT EXISTS {config['stage_name']}
        URL = '{stage_url}'
        CREDENTIALS = (
            AWS_KEY_ID = '{aws_key_id}'
            AWS_SECRET_KEY = '{aws_secret_key}'
        );
    """
    
    # Create file format for CSV
    create_file_format_sql = """
    CREATE FILE FORMAT IF NOT EXISTS jobspy_csv_format
        TYPE = 'CSV'
        FIELD_DELIMITER = ','
        SKIP_HEADER = 1
        NULL_IF = ('NULL', 'null', '', 'None')
        EMPTY_FIELD_AS_NULL = TRUE
        FIELD_OPTIONALLY_ENCLOSED_BY = '"'
        ESCAPE_UNENCLOSED_FIELD = '\\\\';
    """
    
    print(f"Creating stage: {config['stage_name']}")
    print(f"Stage URL: {stage_url}")
    
    # Execute in Snowflake
    snow_hook = SnowflakeHook(snowflake_conn_id=config['snowflake_conn_id'])
    snow_hook.run(create_stage_sql)
    snow_hook.run(create_file_format_sql)
    
    print("Stage and file format created successfully")
    
    return f"Stage {config['stage_name']} ready"


def create_snowflake_table_task(**context):
    """
    Task 4: Create Snowflake table (if not exists)
    - Creates table with all JobSpy columns
    - Uses VARCHAR for most fields for flexibility
    """
    config = get_config()
    
    print(f"Creating Snowflake table: {config['table_name']}")
    
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {config['table_name']} (
        -- Core job info
        site VARCHAR(50),
        job_url VARCHAR(2000),
        job_url_direct VARCHAR(2000),
        title VARCHAR(500),
        company VARCHAR(500),
        location VARCHAR(500),
        job_type VARCHAR(100),
        date_posted DATE,
        
        -- Salary info
        salary_source VARCHAR(100),
        interval VARCHAR(50),
        min_amount FLOAT,
        max_amount FLOAT,
        currency VARCHAR(10),
        
        -- Remote status
        is_remote BOOLEAN,
        
        -- Description
        description TEXT,
        
        -- Company info
        company_url VARCHAR(2000),
        company_url_direct VARCHAR(2000),
        company_addresses TEXT,
        company_industry VARCHAR(500),
        company_num_employees VARCHAR(100),
        company_revenue VARCHAR(100),
        company_description TEXT,
        
        -- Media
        logo_photo_url VARCHAR(2000),
        banner_photo_url VARCHAR(2000),
        ceo_name VARCHAR(200),
        ceo_photo_url VARCHAR(2000),
        
        -- Additional
        emails TEXT,
        job_level VARCHAR(100),
        job_function VARCHAR(200),
        listing_type VARCHAR(100),
        company_rating FLOAT,
        company_reviews_count INTEGER,
        
        -- Metadata
        loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
        load_date DATE DEFAULT CURRENT_DATE()
    );
    """
    
    snow_hook = SnowflakeHook(snowflake_conn_id=config['snowflake_conn_id'])
    snow_hook.run(create_table_sql)
    
    print(f"Table {config['table_name']} created/verified")
    
    return f"Table {config['table_name']} ready"


def load_to_snowflake_task(**context):
    """
    Task 5: Load data from S3 stage into Snowflake
    - Uses COPY INTO command
    - Forces reload to handle re-runs
    """
    config = get_config()
    exec_date = context['ds']
    year, month, day = exec_date.split('-')
    
    # Path pattern in stage
    stage_path = f"year={year}/month={month}/day={day}/"
    
    print(f"Loading data from stage path: {stage_path}")
    
    # Build column list for COPY INTO
    columns = [
        'site', 'job_url', 'job_url_direct', 'title', 'company', 'location',
        'job_type', 'date_posted', 'salary_source', 'interval', 'min_amount',
        'max_amount', 'currency', 'is_remote', 'description', 'company_url',
        'company_url_direct', 'company_addresses', 'company_industry',
        'company_num_employees', 'company_revenue', 'company_description',
        'logo_photo_url', 'banner_photo_url', 'ceo_name', 'ceo_photo_url',
        'emails', 'job_level', 'job_function', 'listing_type',
        'company_rating', 'company_reviews_count'
    ]
    
    columns_str = ', '.join(columns)
    
    # COPY INTO SQL with FORCE = TRUE
    copy_sql = f"""
    COPY INTO {config['table_name']} ({columns_str})
    FROM @{config['stage_name']}/{stage_path}
    FILE_FORMAT = jobspy_csv_format
    ON_ERROR = 'CONTINUE'
    FORCE = TRUE;
    """
    
    print(f"Executing COPY INTO {config['table_name']}")
    
    snow_hook = SnowflakeHook(snowflake_conn_id=config['snowflake_conn_id'])
    snow_hook.run(copy_sql)
    
    # Verify data was loaded
    verify_sql = f"""
    SELECT COUNT(*) as row_count 
    FROM {config['table_name']} 
    WHERE load_date = CURRENT_DATE();
    """
    
    result = snow_hook.get_records(verify_sql)
    row_count = result[0][0] if result else 0
    
    print(f"Rows loaded today: {row_count}")
    
    context['task_instance'].xcom_push(key='rows_loaded', value=row_count)
    
    return f"Loaded {row_count} rows into {config['table_name']}"


def cleanup_task(**context):
    """Task 6: Clean up local temp files"""
    local_file = context['task_instance'].xcom_pull(
        task_ids='scrape_jobs',
        key='local_file'
    )
    
    if local_file and os.path.exists(local_file):
        os.remove(local_file)
        print(f"Deleted: {local_file}")
    
    return "Cleanup complete"


def log_summary_task(**context):
    """Task 7: Log pipeline summary"""
    config = get_config()
    
    job_count = context['task_instance'].xcom_pull(task_ids='scrape_jobs', key='job_count')
    s3_path = context['task_instance'].xcom_pull(task_ids='upload_to_s3', key='s3_path')
    rows_loaded = context['task_instance'].xcom_pull(task_ids='load_to_snowflake', key='rows_loaded')
    
    summary = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                 JOBSPY PIPELINE SUMMARY                      ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Execution Date    : {context['ds']}
    ║  Search Term       : {config['search_term']}
    ║  Location          : {config['location']}
    ║  Jobs Scraped      : {job_count}
    ║  S3 Location       : {s3_path}
    ║  Snowflake Table   : {config['table_name']}
    ║  Rows Loaded       : {rows_loaded}
    ║  Status            : ✅ SUCCESS
    ╚══════════════════════════════════════════════════════════════╝
    """
    
    print(summary)
    return "Pipeline completed successfully"


# =============================================================================
# DAG DEFINITION
# =============================================================================

with DAG(
    dag_id='jobspy_to_snowflake',
    default_args=default_args,
    description='Scrape jobs with JobSpy, upload to S3, and load to Snowflake',
    schedule='@daily',
    catchup=False,
    tags=['jobspy', 's3', 'snowflake', 'etl'],
) as dag:
    
    # Task 1: Scrape jobs
    scrape_jobs = PythonOperator(
        task_id='scrape_jobs',
        python_callable=scrape_jobs_task,
    )
    
    # Task 2: Upload to S3
    upload_to_s3 = PythonOperator(
        task_id='upload_to_s3',
        python_callable=upload_to_s3_task,
    )
    
    # Task 3: Create Snowflake stage
    create_stage = PythonOperator(
        task_id='create_snowflake_stage',
        python_callable=create_snowflake_stage_task,
    )
    
    # Task 4: Create Snowflake table
    create_table = PythonOperator(
        task_id='create_snowflake_table',
        python_callable=create_snowflake_table_task,
    )
    
    # Task 5: Load data into Snowflake
    load_to_snowflake = PythonOperator(
        task_id='load_to_snowflake',
        python_callable=load_to_snowflake_task,
    )
    
    # Task 6: Cleanup
    cleanup = PythonOperator(
        task_id='cleanup',
        python_callable=cleanup_task,
    )
    
    # Task 7: Log summary
    log_summary = PythonOperator(
        task_id='log_summary',
        python_callable=log_summary_task,
    )
    
    # Pipeline flow
    scrape_jobs >> upload_to_s3 >> create_stage >> create_table >> load_to_snowflake >> cleanup >> log_summary