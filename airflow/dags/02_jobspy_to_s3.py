"""
JobSpy to S3 DAG
===================================
This DAG demonstrates:
1. Using Airflow Variables for configuration
2. Scraping jobs using JobSpy library
3. Uploading to AWS S3
4. Clean separation of config and code
"""

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
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
}

# =============================================================================
# TASK FUNCTIONS
# =============================================================================

def scrape_jobs_task(**context):
    """
    Task 1: Scrape jobs using JobSpy
    - Reads config from Airflow Variables
    - Fetches job listings from multiple sites
    - Saves to local CSV
    """
    from jobspy import scrape_jobs
    
    # =========================================================================
    # FETCH CONFIG FROM AIRFLOW VARIABLES
    # =========================================================================
    
    # Method 1: Get individual variables with defaults
    search_term = Variable.get("jobspy_search_term", default_var="software engineer")
    location = Variable.get("jobspy_location", default_var="San Francisco, CA")
    results_wanted = int(Variable.get("jobspy_results_wanted", default_var="20"))
    hours_old = int(Variable.get("jobspy_hours_old", default_var="72"))
    country = Variable.get("jobspy_country", default_var="USA")
    
    # Method 2: Get sites as JSON list
    sites_json = Variable.get("jobspy_sites", default_var='["indeed", "linkedin", "google"]')
    sites = json.loads(sites_json)
    
    exec_date = context['ds']
    
    print(f"=== Configuration from Airflow Variables ===")
    print(f"Search Term:     {search_term}")
    print(f"Location:        {location}")
    print(f"Sites:           {sites}")
    print(f"Results Wanted:  {results_wanted}")
    print(f"Hours Old:       {hours_old}")
    print(f"============================================")
    
    # Scrape jobs
    jobs = scrape_jobs(
        site_name=sites,
        search_term=search_term,
        google_search_term=f"{search_term} jobs near {location} since yesterday",
        location=location,
        results_wanted=results_wanted,
        hours_old=hours_old,
        country_indeed=country,
    )
    
    print(f"Found {len(jobs)} jobs")
    
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
    
    return f"Scraped {len(jobs)} jobs"


def upload_to_s3_task(**context):
    """
    Task 2: Upload CSV to S3
    - Reads S3 config from Airflow Variables
    - Uploads with date partitioning
    """
    # =========================================================================
    # FETCH S3 CONFIG FROM AIRFLOW VARIABLES
    # =========================================================================
    s3_bucket = Variable.get("jobspy_s3_bucket")  # Required - no default
    s3_prefix = Variable.get("jobspy_s3_prefix", default_var="jobs/raw/")
    aws_conn_id = Variable.get("jobspy_aws_conn_id", default_var="aws_default")
    
    # Pull from XCom
    local_file = context['task_instance'].xcom_pull(
        task_ids='scrape_jobs',
        key='local_file'
    )
    
    exec_date = context['ds']
    year, month, day = exec_date.split('-')
    
    # Create S3 key with date partitioning
    s3_key = f"{s3_prefix}year={year}/month={month}/day={day}/jobs_{exec_date}.csv"
    
    print(f"Uploading to s3://{s3_bucket}/{s3_key}")
    
    # Upload using S3Hook
    s3_hook = S3Hook(aws_conn_id=aws_conn_id)
    s3_hook.load_file(
        filename=local_file,
        key=s3_key,
        bucket_name=s3_bucket,
        replace=True
    )
    
    s3_path = f"s3://{s3_bucket}/{s3_key}"
    context['task_instance'].xcom_push(key='s3_path', value=s3_path)
    
    return f"Uploaded to {s3_path}"


def cleanup_task(**context):
    """Task 3: Clean up local temp files"""
    local_file = context['task_instance'].xcom_pull(
        task_ids='scrape_jobs',
        key='local_file'
    )
    
    if local_file and os.path.exists(local_file):
        os.remove(local_file)
        print(f"Deleted: {local_file}")
    
    return "Cleanup complete"


def log_summary_task(**context):
    """Task 4: Log pipeline summary"""
    job_count = context['task_instance'].xcom_pull(task_ids='scrape_jobs', key='job_count')
    s3_path = context['task_instance'].xcom_pull(task_ids='upload_to_s3', key='s3_path')
    
    # Also log the config that was used
    search_term = Variable.get("jobspy_search_term", default_var="software engineer")
    location = Variable.get("jobspy_location", default_var="San Francisco, CA")
    
    summary = f"""
    ========== PIPELINE SUMMARY ==========
    Execution Date: {context['ds']}
    Search Term:    {search_term}
    Location:       {location}
    Jobs Scraped:   {job_count}
    S3 Location:    {s3_path}
    Status:         SUCCESS
    ======================================
    """
    print(summary)
    return summary


# =============================================================================
# DAG DEFINITION
# =============================================================================

with DAG(
    dag_id='jobspy_to_s3',
    default_args=default_args,
    description='Scrape jobs with JobSpy and upload to S3 (Variable-driven)',
    schedule='@daily',
    catchup=False,
    tags=['jobspy', 's3', 'etl'],
) as dag:
    
    scrape_jobs = PythonOperator(
        task_id='scrape_jobs',
        python_callable=scrape_jobs_task,
    )
    
    upload_to_s3 = PythonOperator(
        task_id='upload_to_s3',
        python_callable=upload_to_s3_task,
    )
    
    cleanup = PythonOperator(
        task_id='cleanup',
        python_callable=cleanup_task,
    )
    
    log_summary = PythonOperator(
        task_id='log_summary',
        python_callable=log_summary_task,
    )
    
    # Pipeline flow
    scrape_jobs >> upload_to_s3 >> cleanup >> log_summary