"""dataseeding

Revision ID: 0ee619f3516b
Revises: dadf308f3867
Create Date: 2026-01-21 19:24:23.766065

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0ee619f3516b'
down_revision: Union[str, Sequence[str], None] = 'dadf308f3867'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("""
               -- Seed PE Org-AI-R Sectors
INSERT INTO focus_groups (focus_group_id, platform, group_name, group_code, display_order) VALUES
    ('pe_manufacturing', 'pe_org_air', 'Manufacturing', 'MFG', 1),
    ('pe_financial_services', 'pe_org_air', 'Financial Services', 'FIN', 2),
    ('pe_healthcare', 'pe_org_air', 'Healthcare', 'HC', 3),
    ('pe_technology', 'pe_org_air', 'Technology', 'TECH', 4),
    ('pe_retail', 'pe_org_air', 'Retail & Consumer', 'RTL', 5),
    ('pe_energy', 'pe_org_air', 'Energy & Utilities', 'ENR', 6),
    ('pe_professional_services', 'pe_org_air', 'Professional Services', 'PS', 7);

-- Seed Dimensions
INSERT INTO dimensions (dimension_id, platform, dimension_name, dimension_code, display_order) VALUES
    ('pe_dim_data_infra', 'pe_org_air', 'Data Infrastructure', 'data_infrastructure', 1),
    ('pe_dim_governance', 'pe_org_air', 'AI Governance', 'ai_governance', 2),
    ('pe_dim_tech_stack', 'pe_org_air', 'Technology Stack', 'technology_stack', 3),
    ('pe_dim_talent', 'pe_org_air', 'Talent', 'talent', 4),
    ('pe_dim_leadership', 'pe_org_air', 'Leadership', 'leadership', 5),
    ('pe_dim_use_cases', 'pe_org_air', 'Use Case Portfolio', 'use_case_portfolio', 6),
    ('pe_dim_culture', 'pe_org_air', 'Culture', 'culture', 7);
    -- Manufacturing (emphasis: data infra, tech stack, use cases)
INSERT INTO focus_group_dimension_weights (focus_group_id, dimension_id, weight,
weight_rationale) VALUES
('pe_manufacturing', 'pe_dim_data_infra', 0.22, 'OT/IT integration critical'),
('pe_manufacturing', 'pe_dim_governance', 0.12, 'Less regulatory than finance/health'),
('pe_manufacturing', 'pe_dim_tech_stack', 0.18, 'Edge computing, IoT platforms'),
('pe_manufacturing', 'pe_dim_talent', 0.15, 'AI + manufacturing expertise scarce'),
('pe_manufacturing', 'pe_dim_leadership', 0.12, 'Traditional leadership acceptable'),
('pe_manufacturing', 'pe_dim_use_cases', 0.14, 'Clear ROI in operations'),
('pe_manufacturing', 'pe_dim_culture', 0.07, 'Safety culture > innovation');

-- Financial Services (emphasis: governance, talent)
INSERT INTO focus_group_dimension_weights (focus_group_id, dimension_id, weight,
weight_rationale) VALUES
('pe_financial_services', 'pe_dim_data_infra', 0.16, 'Mature infrastructure'),
('pe_financial_services', 'pe_dim_governance', 0.22, 'Regulatory imperative'),
('pe_financial_services', 'pe_dim_tech_stack', 0.14, 'Standard cloud stacks'),
('pe_financial_services', 'pe_dim_talent', 0.18, 'Quant + ML talent critical'),
('pe_financial_services', 'pe_dim_leadership', 0.12, 'C-suite AI awareness high'),
('pe_financial_services', 'pe_dim_use_cases', 0.10, 'Well-understood use cases'),
('pe_financial_services', 'pe_dim_culture', 0.08, 'Risk-averse by design');

-- Healthcare (emphasis: governance, data, leadership)
INSERT INTO focus_group_dimension_weights (focus_group_id, dimension_id, weight,
weight_rationale) VALUES
('pe_healthcare', 'pe_dim_data_infra', 0.20, 'EHR integration critical'),
('pe_healthcare', 'pe_dim_governance', 0.20, 'FDA/HIPAA compliance'),
('pe_healthcare', 'pe_dim_tech_stack', 0.14, 'EHR-centric ecosystems'),
('pe_healthcare', 'pe_dim_talent', 0.15, 'Clinical + AI dual expertise'),
('pe_healthcare', 'pe_dim_leadership', 0.15, 'Physician champions matter'),
('pe_healthcare', 'pe_dim_use_cases', 0.10, 'Long validation cycles'),
('pe_healthcare', 'pe_dim_culture', 0.06, 'Evidence-based culture exists');

-- Technology (emphasis: talent, tech stack, use cases)
INSERT INTO focus_group_dimension_weights (focus_group_id, dimension_id, weight,
weight_rationale) VALUES
('pe_technology', 'pe_dim_data_infra', 0.15, 'Assumed competent'),
('pe_technology', 'pe_dim_governance', 0.12, 'Less regulated'),
('pe_technology', 'pe_dim_tech_stack', 0.18, 'Core differentiator'),
('pe_technology', 'pe_dim_talent', 0.22, 'Talent is everything'),
('pe_technology', 'pe_dim_leadership', 0.13, 'Tech-savvy by default'),
('pe_technology', 'pe_dim_use_cases', 0.15, 'Product innovation'),
('pe_technology', 'pe_dim_culture', 0.05, 'Innovation assumed');

-- Retail & Consumer (emphasis: data, use cases)
INSERT INTO focus_group_dimension_weights (focus_group_id, dimension_id, weight,
weight_rationale) VALUES
('pe_retail', 'pe_dim_data_infra', 0.20, 'Customer data platforms'),
('pe_retail', 'pe_dim_governance', 0.12, 'Privacy focus, less regulated'),
('pe_retail', 'pe_dim_tech_stack', 0.15, 'Standard cloud + CDP'),
('pe_retail', 'pe_dim_talent', 0.15, 'Data science accessible'),
('pe_retail', 'pe_dim_leadership', 0.13, 'Digital transformation focus'),
('pe_retail', 'pe_dim_use_cases', 0.18, 'Clear revenue impact'),
('pe_retail', 'pe_dim_culture', 0.07, 'Customer-centric exists');

-- Energy & Utilities (emphasis: data, tech stack, use cases)
INSERT INTO focus_group_dimension_weights (focus_group_id, dimension_id, weight,
weight_rationale) VALUES
('pe_energy', 'pe_dim_data_infra', 0.22, 'SCADA/OT data critical'),
('pe_energy', 'pe_dim_governance', 0.15, 'Regulatory + safety'),
('pe_energy', 'pe_dim_tech_stack', 0.18, 'Grid tech, edge computing'),
('pe_energy', 'pe_dim_talent', 0.12, 'Talent scarcity'),
('pe_energy', 'pe_dim_leadership', 0.13, 'Traditional but evolving'),
('pe_energy', 'pe_dim_use_cases', 0.15, 'Clear operational value'),
('pe_energy', 'pe_dim_culture', 0.05, 'Safety culture paramount');

-- Professional Services (emphasis: talent, leadership)
INSERT INTO focus_group_dimension_weights (focus_group_id, dimension_id, weight,
weight_rationale) VALUES
('pe_professional_services', 'pe_dim_data_infra', 0.14, 'Document-centric'),
('pe_professional_services', 'pe_dim_governance', 0.15, 'Client confidentiality'),
('pe_professional_services', 'pe_dim_tech_stack', 0.12, 'Standard productivity'),
('pe_professional_services', 'pe_dim_talent', 0.22, 'People are the product'),
('pe_professional_services', 'pe_dim_leadership', 0.17, 'Partner adoption critical'),
('pe_professional_services', 'pe_dim_use_cases', 0.12, 'Client + internal'),
('pe_professional_services', 'pe_dim_culture', 0.08, 'Innovation varies');
               
               -- migrations/versions/002c_seed_calibrations.sql
INSERT INTO focus_group_calibrations (focus_group_id, parameter_name, parameter_value,
parameter_type, description) VALUES
-- Manufacturing
('pe_manufacturing', 'h_r_baseline', 72, 'numeric', 'Systematic opportunity baseline'),
('pe_manufacturing', 'ebitda_multiplier', 0.90, 'numeric', 'Conservative EBITDA attribution'),
('pe_manufacturing', 'talent_concentration_threshold', 0.20, 'threshold', 'Lower due to talent scarcity'),
('pe_manufacturing', 'position_factor_delta', 0.15, 'numeric', 'H^R position adjustment'),
-- Financial Services
('pe_financial_services', 'h_r_baseline', 82, 'numeric', 'Higher due to data maturity'),
('pe_financial_services', 'ebitda_multiplier', 1.10, 'numeric', 'Higher AI leverage'),
('pe_financial_services', 'talent_concentration_threshold', 0.25, 'threshold', 'Standard threshold'),
('pe_financial_services', 'position_factor_delta', 0.15, 'numeric', 'H^R position adjustment'),
('pe_financial_services', 'governance_minimum', 60, 'threshold', 'Min governance for approval'),
-- Healthcare
('pe_healthcare', 'h_r_baseline', 78, 'numeric', 'Moderate with growth potential'),
('pe_healthcare', 'ebitda_multiplier', 1.00, 'numeric', 'Standard attribution'),
('pe_healthcare', 'talent_concentration_threshold', 0.25, 'threshold', 'Standard threshold'),
('pe_healthcare', 'position_factor_delta', 0.15, 'numeric', 'H^R position adjustment'),
('pe_healthcare', 'governance_minimum', 65, 'threshold', 'Higher governance requirement'),
-- Technology
('pe_technology', 'h_r_baseline', 85, 'numeric', 'Highest - AI native'),
('pe_technology', 'ebitda_multiplier', 1.15, 'numeric', 'Strong AI leverage'),
('pe_technology', 'talent_concentration_threshold', 0.30, 'threshold', 'Higher talent expected'),
('pe_technology', 'position_factor_delta', 0.15, 'numeric', 'H^R position adjustment'),
-- Retail
('pe_retail', 'h_r_baseline', 75, 'numeric', 'Growing AI adoption'),
('pe_retail', 'ebitda_multiplier', 1.05, 'numeric', 'Clear personalization ROI'),
('pe_retail', 'talent_concentration_threshold', 0.25, 'threshold', 'Standard threshold'),
('pe_retail', 'position_factor_delta', 0.15, 'numeric', 'H^R position adjustment'),
-- Energy
('pe_energy', 'h_r_baseline', 68, 'numeric', 'Lower but high potential'),
('pe_energy', 'ebitda_multiplier', 0.85, 'numeric', 'Longer payback periods'),
('pe_energy', 'talent_concentration_threshold', 0.20, 'threshold', 'Lower due to scarcity'),
('pe_energy', 'position_factor_delta', 0.15, 'numeric', 'H^R position adjustment'),
-- Professional Services
('pe_professional_services', 'h_r_baseline', 76, 'numeric', 'Knowledge work automation'),
('pe_professional_services', 'ebitda_multiplier', 1.00, 'numeric', 'Standard attribution'),
('pe_professional_services', 'talent_concentration_threshold', 0.25, 'threshold', 'Standard threshold'),
('pe_professional_services', 'position_factor_delta', 0.15, 'numeric', 'H^R position adjustment');
                """)


def downgrade() -> None:
    """Downgrade schema."""
    pass
