# News sanity eval (per-article TPB effects, isolation runs)

Agents per cell: 20 (first 20 of the shared seeded pool) · weeks: 2 (exposure at week 1) · corpus: news_corpus_qwen.json · deltas are t=0 -> week 2, net of the no-news control cell.

Control drift (raw): att +0.11 norm +0.00 pbc +0.06 E[int] -0.04

| article | type | net Δatt | net Δnorm | net Δpbc | net ΔE[int] | top | expected | match |
|---|---|---|---|---|---|---|---|---|
| preschool_and_infant_care_subsidies__family_impact_1 | family_impact | +0.16 | +0.01 | +0.33 | +0.03 | pbc | pbc | yes |
| preschool_and_infant_care_subsidies__explainer_2 | explainer | +0.13 | +0.03 | +0.30 | -0.01 | pbc | pbc | yes |
| preschool_and_infant_care_subsidies__explainer_1 | explainer | +0.15 | +0.03 | +0.30 | +0.01 | pbc | pbc | yes |
| preschool_and_infant_care_subsidies__announcement_1 | announcement | +0.16 | +0.03 | +0.30 | +0.00 | pbc | pbc | yes |
| infant_childminding_pilot__family_impact_2 | family_impact | +0.12 | +0.02 | +0.29 | -0.00 | pbc | pbc | yes |
| preschool_and_infant_care_subsidies__roundup_1 | roundup | +0.11 | +0.04 | +0.28 | -0.03 | pbc | pbc | yes |
| infant_childminding_pilot__announcement_1 | announcement | +0.11 | +0.05 | +0.28 | +0.04 | pbc | pbc | yes |
| infant_childminding_pilot__explainer_2 | explainer | +0.10 | +0.01 | +0.28 | +0.09 | pbc | pbc | yes |
| flexible_work_arrangement_request_guidelines__explainer_1 | explainer | +0.00 | +0.05 | +0.26 | -0.01 | pbc | pbc/attitude | yes |
| flexible_work_arrangement_request_guidelines__family_impact_2 | family_impact | +0.13 | +0.04 | +0.26 | -0.02 | pbc | pbc/attitude | yes |
| flexible_work_arrangement_request_guidelines__roundup_1 | roundup | +0.00 | +0.01 | +0.26 | -0.02 | pbc | pbc/attitude | yes |
| infant_childminding_pilot__family_impact_1 | family_impact | +0.11 | +0.02 | +0.26 | +0.04 | pbc | pbc | yes |
| infant_childminding_pilot__roundup_1 | roundup | +0.14 | +0.02 | +0.26 | +0.03 | pbc | pbc | yes |
| infant_childminding_pilot__explainer_1 | explainer | +0.10 | +0.01 | +0.26 | +0.04 | pbc | pbc | yes |
| baby_bonus_and_child_development_account__announcement_1 | announcement | +0.19 | +0.02 | +0.26 | +0.03 | pbc | pbc | yes |
| flexible_work_arrangement_request_guidelines__family_impact_1 | family_impact | +0.14 | +0.07 | +0.25 | +0.02 | pbc | pbc/attitude | yes |
| flexible_work_arrangement_request_guidelines__explainer_2 | explainer | +0.02 | +0.02 | +0.25 | -0.03 | pbc | pbc/attitude | yes |
| baby_bonus_and_child_development_account__explainer_2 | explainer | +0.16 | +0.01 | +0.24 | -0.01 | pbc | pbc | yes |
| flexible_work_arrangement_request_guidelines__announcement_1 | announcement | -0.01 | +0.04 | +0.23 | -0.02 | pbc | pbc/attitude | yes |
| shared_parental_leave__explainer_1 | explainer | +0.10 | +0.03 | +0.22 | +0.01 | pbc | pbc/attitude | yes |
| shared_parental_leave__family_impact_2 | family_impact | +0.12 | +0.08 | +0.21 | +0.08 | pbc | pbc/attitude | yes |
| large_family_scheme__family_impact_1 | family_impact | +0.16 | +0.01 | +0.21 | -0.01 | pbc | pbc | yes |
| shared_parental_leave__family_impact_1 | family_impact | +0.15 | +0.09 | +0.21 | +0.03 | pbc | pbc/attitude | yes |
| large_family_scheme__family_impact_2 | family_impact | +0.17 | +0.00 | +0.21 | +0.01 | pbc | pbc | yes |
| large_family_scheme__roundup_1 | roundup | +0.13 | +0.00 | +0.20 | -0.04 | pbc | pbc | yes |
| shared_parental_leave__announcement_1 | announcement | +0.15 | +0.11 | +0.20 | +0.03 | pbc | pbc/attitude | yes |
| shared_parental_leave__roundup_1 | roundup | +0.12 | +0.04 | +0.20 | +0.00 | pbc | pbc/attitude | yes |
| baby_bonus_and_child_development_account__explainer_1 | explainer | +0.16 | +0.01 | +0.20 | +0.01 | pbc | pbc | yes |
| large_family_scheme__announcement_1 | announcement | +0.12 | +0.02 | +0.20 | -0.01 | pbc | pbc | yes |
| large_family_scheme__explainer_1 | explainer | +0.14 | +0.04 | +0.19 | -0.01 | pbc | pbc | yes |
| shared_parental_leave__explainer_2 | explainer | +0.10 | +0.05 | +0.18 | -0.01 | pbc | pbc/attitude | yes |
| enhanced_paternity_leave__explainer_2 | explainer | +0.12 | +0.10 | +0.16 | -0.05 | pbc | pbc/attitude | yes |
| enhanced_paternity_leave__roundup_1 | roundup | +0.10 | +0.16 | +0.15 | -0.02 | norm | pbc/attitude | **NO** |
| child_lifesg_credits__roundup_1 | roundup | +0.09 | +0.01 | +0.15 | -0.04 | pbc | pbc | yes |
| child_lifesg_credits__announcement_1 | announcement | +0.09 | +0.02 | +0.15 | -0.02 | pbc | pbc | yes |
| enhanced_paternity_leave__explainer_1 | explainer | +0.08 | +0.13 | +0.15 | +0.03 | pbc | pbc/attitude | yes |
| child_lifesg_credits__family_impact_1 | family_impact | +0.08 | +0.01 | +0.15 | -0.03 | pbc | pbc | yes |
| large_family_scheme__explainer_2 | explainer | +0.11 | +0.02 | +0.15 | +0.00 | pbc | pbc | yes |
| child_lifesg_credits__family_impact_2 | family_impact | +0.12 | +0.02 | +0.15 | -0.03 | pbc | pbc | yes |
| enhanced_paternity_leave__family_impact_2 | family_impact | +0.14 | +0.11 | +0.15 | +0.00 | pbc | pbc/attitude | yes |
| child_lifesg_credits__explainer_1 | explainer | +0.11 | +0.01 | +0.14 | +0.01 | pbc | pbc | yes |
| enhanced_paternity_leave__announcement_1 | announcement | +0.11 | +0.12 | +0.14 | -0.01 | pbc | pbc/attitude | yes |
| enhanced_paternity_leave__family_impact_1 | family_impact | +0.11 | +0.13 | +0.12 | -0.02 | norm | pbc/attitude | **NO** |
| child_lifesg_credits__explainer_2 | explainer | +0.07 | +0.01 | +0.12 | -0.05 | pbc | pbc | yes |
| baby_bonus_and_child_development_account__family_impact_1 | family_impact | +0.11 | +0.03 | +0.12 | -0.03 | pbc | pbc | yes |

## Mean net delta by article type

| article type | n | Δatt | Δnorm | Δpbc | ΔE[int] |
|---|---|---|---|---|---|
| announcement | 8 | +0.11 | +0.05 | +0.22 | +0.01 |
| explainer | 16 | +0.10 | +0.04 | +0.21 | +0.00 |
| family_impact | 14 | +0.13 | +0.05 | +0.21 | +0.00 |
| roundup | 7 | +0.10 | +0.04 | +0.22 | -0.02 |

## Mean net delta by policy

| policy | n | Δatt | Δnorm | Δpbc | ΔE[int] |
|---|---|---|---|---|---|
| Baby Bonus & Child Development Account | 4 | +0.16 | +0.02 | +0.21 | +0.00 |
| Child LifeSG Credits | 6 | +0.09 | +0.01 | +0.14 | -0.03 |
| Enhanced Paternity Leave | 6 | +0.11 | +0.13 | +0.14 | -0.01 |
| Flexible Work Arrangement Request Guidelines | 6 | +0.04 | +0.04 | +0.26 | -0.02 |
| Infant Childminding Pilot | 6 | +0.12 | +0.02 | +0.27 | +0.04 |
| Large Family Scheme | 6 | +0.14 | +0.02 | +0.19 | -0.01 |
| Preschool & Infant Care Subsidies | 5 | +0.14 | +0.03 | +0.30 | +0.00 |
| Shared Parental Leave | 6 | +0.12 | +0.07 | +0.21 | +0.02 |
