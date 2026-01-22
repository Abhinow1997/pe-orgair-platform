"""creation base tables

Revision ID: dadf308f3867
Revises: ffa7dce781be
Create Date: 2026-01-21 19:17:54.816320

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'dadf308f3867'
down_revision: Union[str, Sequence[str], None] = 'ffa7dce781be'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("""
                   
    -- ============================================
-- FOCUS GROUPS MASTER TABLE
-- ============================================
CREATE TABLE focus_groups (
    focus_group_id VARCHAR(50) PRIMARY KEY,
    platform VARCHAR(20) NOT NULL CHECK (platform IN ('pe_org_air', 'individual_air')),
    group_name VARCHAR(100) NOT NULL,
    group_code VARCHAR(30) NOT NULL,
    group_description TEXT,
    display_order INTEGER NOT NULL,
    icon_name VARCHAR(50),
    color_hex VARCHAR(7),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (platform, group_code)
);

-- ============================================
-- DIMENSIONS TABLE (PE Org-AI-R: 7 Dimensions)
-- ============================================
CREATE TABLE dimensions (
    dimension_id VARCHAR(50) PRIMARY KEY,
    platform VARCHAR(20) NOT NULL,
    dimension_name VARCHAR(100) NOT NULL,
    dimension_code VARCHAR(50) NOT NULL,
    description TEXT,
    min_score DECIMAL(5,2) DEFAULT 0,
    max_score DECIMAL(5,2) DEFAULT 100,
    display_order INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (platform, dimension_code)
);

-- ============================================
-- FOCUS GROUP DIMENSION WEIGHTS
-- ============================================
CREATE TABLE focus_group_dimension_weights (
    weight_id SERIAL PRIMARY KEY,
    focus_group_id VARCHAR(50) NOT NULL REFERENCES focus_groups(focus_group_id),
    dimension_id VARCHAR(50) NOT NULL REFERENCES dimensions(dimension_id),
    weight DECIMAL(4,3) NOT NULL CHECK (weight >= 0 AND weight <= 1),
    weight_rationale TEXT,
    effective_from DATE NOT NULL DEFAULT CURRENT_DATE,
    effective_to DATE,
    is_current BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (focus_group_id, dimension_id, effective_from)
);

CREATE INDEX idx_weights_current ON focus_group_dimension_weights(focus_group_id, is_current)
    WHERE is_current = TRUE;

-- ============================================
-- FOCUS GROUP CALIBRATIONS
-- ============================================
CREATE TABLE focus_group_calibrations (
    calibration_id SERIAL PRIMARY KEY,
    focus_group_id VARCHAR(50) NOT NULL REFERENCES focus_groups(focus_group_id),
    parameter_name VARCHAR(100) NOT NULL,
    parameter_value DECIMAL(10,4) NOT NULL,
    parameter_type VARCHAR(20) DEFAULT 'numeric',
    description TEXT,
    effective_from DATE NOT NULL DEFAULT CURRENT_DATE,
    effective_to DATE,
    is_current BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (focus_group_id, parameter_name, effective_from)
);

                   CREATE TABLE organizations (
    organization_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Identity
    legal_name VARCHAR(255) NOT NULL,
    display_name VARCHAR(255),
    ticker_symbol VARCHAR(10),
    cik_number VARCHAR(20),
    duns_number VARCHAR(20),

    -- Sector Assignment
    focus_group_id VARCHAR(50) NOT NULL REFERENCES focus_groups(focus_group_id),

    -- Industry Classification
    primary_sic_code VARCHAR(10),
    primary_naics_code VARCHAR(10),

    -- Firmographics
    employee_count INTEGER,
    annual_revenue_usd DECIMAL(15,2),
    founding_year INTEGER,
    headquarters_country VARCHAR(3),
    headquarters_state VARCHAR(50),
    headquarters_city VARCHAR(100),
    website_url VARCHAR(500),

    -- Status
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'archived')),

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),

    CONSTRAINT chk_org_pe_platform CHECK (focus_group_id LIKE 'pe_%')
);

CREATE INDEX idx_org_focus_group ON organizations(focus_group_id);
CREATE INDEX idx_org_ticker ON organizations(ticker_symbol) WHERE ticker_symbol IS NOT NULL;
                   
-- Manufacturing Sector Attributes
CREATE TABLE org_attributes_manufacturing (
    organization_id UUID PRIMARY KEY REFERENCES organizations(organization_id),
    ot_systems VARCHAR(100)[],
    it_ot_integration VARCHAR(20),
    scada_vendor VARCHAR(100),
    mes_system VARCHAR(100),
    plant_count INTEGER,
    automation_level VARCHAR(20),
    iot_platforms VARCHAR(100)[],
    digital_twin_status VARCHAR(20),
    edge_computing BOOLEAN DEFAULT FALSE,
    supply_chain_visibility VARCHAR(20),
    demand_forecasting_ai BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Financial Services Sector Attributes
CREATE TABLE org_attributes_financial_services (
    organization_id UUID PRIMARY KEY REFERENCES organizations(organization_id),
    regulatory_bodies VARCHAR(50)[],
    charter_type VARCHAR(50),
    model_risk_framework VARCHAR(50),
    mrm_team_size INTEGER,
    model_inventory_count INTEGER,
    algo_trading BOOLEAN DEFAULT FALSE,
    fraud_detection_ai BOOLEAN DEFAULT FALSE,
    credit_ai BOOLEAN DEFAULT FALSE,
    aml_ai BOOLEAN DEFAULT FALSE,
    aum_billions DECIMAL(12,2),
    total_assets_billions DECIMAL(12,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Healthcare Sector Attributes
CREATE TABLE org_attributes_healthcare (
    organization_id UUID PRIMARY KEY REFERENCES organizations(organization_id),
    hipaa_certified BOOLEAN DEFAULT FALSE,
    hitrust_certified BOOLEAN DEFAULT FALSE,
    fda_clearances VARCHAR(100)[],
    fda_clearance_count INTEGER DEFAULT 0,
    ehr_system VARCHAR(100),
    ehr_integration_level VARCHAR(20),
    fhir_enabled BOOLEAN DEFAULT FALSE,
    clinical_ai_deployed BOOLEAN DEFAULT FALSE,
    imaging_ai BOOLEAN DEFAULT FALSE,
    org_type VARCHAR(50),
    bed_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Technology Sector Attributes
CREATE TABLE org_attributes_technology (
    organization_id UUID PRIMARY KEY REFERENCES organizations(organization_id),
    tech_category VARCHAR(50),
    primary_language VARCHAR(50),
    cloud_native BOOLEAN DEFAULT TRUE,
    github_org VARCHAR(100),
    github_stars_total INTEGER,
    open_source_projects INTEGER,
    ml_platform VARCHAR(100),
    llm_integration BOOLEAN DEFAULT FALSE,
    ai_product_features INTEGER,
    gpu_infrastructure BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Retail Sector Attributes
CREATE TABLE org_attributes_retail (
    organization_id UUID PRIMARY KEY REFERENCES organizations(organization_id),
    retail_type VARCHAR(50),
    store_count INTEGER,
    ecommerce_pct DECIMAL(5,2),
    cdp_vendor VARCHAR(100),
    loyalty_program BOOLEAN DEFAULT FALSE,
    loyalty_members INTEGER,
    personalization_ai BOOLEAN DEFAULT FALSE,
    recommendation_engine VARCHAR(100),
    demand_forecasting BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Energy Sector Attributes
CREATE TABLE org_attributes_energy (
    organization_id UUID PRIMARY KEY REFERENCES organizations(organization_id),
    energy_type VARCHAR(50),
    regulated BOOLEAN DEFAULT FALSE,
    scada_systems VARCHAR(100)[],
    ami_deployed BOOLEAN DEFAULT FALSE,
    smart_grid_pct DECIMAL(5,2),
    generation_capacity_mw DECIMAL(12,2),
    grid_optimization_ai BOOLEAN DEFAULT FALSE,
    predictive_maintenance BOOLEAN DEFAULT FALSE,
    renewable_pct DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Professional Services Sector Attributes
CREATE TABLE org_attributes_professional_services (
    organization_id UUID PRIMARY KEY REFERENCES organizations(organization_id),
    firm_type VARCHAR(50),
    partnership_model VARCHAR(50),
    partner_count INTEGER,
    professional_staff INTEGER,
    km_system VARCHAR(100),
    document_ai BOOLEAN DEFAULT FALSE,
    knowledge_graph BOOLEAN DEFAULT FALSE,
    client_ai_services BOOLEAN DEFAULT FALSE,
    internal_ai_tools BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("""
    DROP TABLE IF EXISTS org_attributes_professional_services;
    DROP TABLE IF EXISTS org_attributes_energy;
    DROP TABLE IF EXISTS org_attributes_retail;
    DROP TABLE IF EXISTS org_attributes_technology;
    DROP TABLE IF EXISTS org_attributes_healthcare;
    DROP TABLE IF EXISTS org_attributes_financial_services;
    DROP TABLE IF EXISTS org_attributes_manufacturing;
    DROP TABLE IF EXISTS organizations;
    DROP TABLE IF EXISTS focus_group_calibrations;
    DROP TABLE IF EXISTS focus_group_dimension_weights;
    DROP TABLE IF EXISTS dimensions;
    DROP TABLE IF EXISTS focus_groups;
    """)
