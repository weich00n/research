# Social network inspection — social_network_qwen.json

Agents (profile file): 100 | network entries: 100
Valid directed edges (A follows B): 824 | density 0.083

## Degree
- out-degree (friends chosen): mean 8.2, median 8, min 3, max 22 | agents following nobody: 0
- in-degree (followers):       mean 8.2, median 7, min 0, max 27 | agents nobody follows: 1

## Reciprocity
- mutual follows: 214/824 edges (26%) are reciprocated

## Homophily (do friends resemble each other more than chance?)
| attribute | edge same-rate | random baseline | lift |
|---|---|---|---|
| planning_area | 0.313 | 0.050 | 6.22× |
| relationship_status | 0.465 | 0.374 | 1.24× |
| gender | 0.562 | 0.496 | 1.13× |
| education | 0.504 | 0.377 | 1.34× |

- age gap |Δ|: 6.5 yrs over friendships vs 7.4 yrs for random pairs (younger gap = age homophily)

## Popularity (most-followed)
- agent_027 (35M, Married, Professional, Bedok): 27 followers
- agent_084 (44M, Married, Senior Official or Manager, Tampines): 24 followers
- agent_007 (27F, Single, Senior Official or Manager, Tampines): 23 followers
- agent_090 (33M, Dating, Senior Official or Manager, Woodlands): 21 followers
- agent_047 (33F, Single, Professional, Yishun): 20 followers

## Sample friend lists

**agent_001** — 39F, Married, Unemployed, Ang Mo Kio → follows 9:
  - agent_059: 27F, Single, Associate Professional or Technician, Tampines
  - agent_066: 39M, Married, Professional, Ang Mo Kio
  - agent_035: 24F, Dating, Student, Bedok
  - agent_006: 42M, Married, Professional, Bukit Timah
  - agent_088: 32M, Dating, Professional, Novena
  - agent_094: 21M, Dating, Service or Sales Worker, Bukit Panjang
  - agent_007: 27F, Single, Senior Official or Manager, Tampines
  - agent_033: 40F, Married, Professional, Pasir Ris

**agent_050** — 44M, Married, Senior Official or Manager, Serangoon → follows 10:
  - agent_013: 35M, Married, Professional, Punggol
  - agent_026: 32F, Single, Senior Official or Manager, Yishun
  - agent_028: 40M, Married, Professional, Sengkang
  - agent_035: 24F, Dating, Student, Bedok
  - agent_047: 33F, Single, Professional, Yishun
  - agent_058: 34F, Married, Professional, Bedok
  - agent_075: 40M, Dating, Professional, Toa Payoh
  - agent_084: 44M, Married, Senior Official or Manager, Tampines

**agent_100** — 26F, Married, Senior Official or Manager, Pasir Ris → follows 5:
  - agent_027: 35M, Married, Professional, Bedok
  - agent_033: 40F, Married, Professional, Pasir Ris
  - agent_026: 32F, Single, Senior Official or Manager, Yishun
  - agent_008: 33M, Married, Associate Professional or Technician, Bishan
  - agent_048: 24M, Dating, Student, Woodlands