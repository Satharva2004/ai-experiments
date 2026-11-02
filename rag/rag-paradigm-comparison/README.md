## Summary of Results

Below is a minimal summary of evaluation metrics across retrieval paradigms on enterprise document domains:

| Category / Domain | Best Performing Paradigm | Avg Judge Score (1–5) | Retrieval Hit Rate | Key Performance Insight |
|---|---|---|---|---|
| **Structured / Financial** | Hybrid RAG (RRF) | **4.90** | **100.0%** | High agreement between dense vector and BM25 |
| **Unstructured / Marketing** | Vector RAG (Pinecone) | **4.20** | **60.0%** | Superior semantic capture over keyword matching |
| **Paraphrased / Product QA** | Vector RAG (Pinecone) | **2.45** | **40.0%** | Resilience against user query vocabulary shifts |
| **Templated / Service** | Hybrid RAG (RRF) | **4.25** | **10.0%** | High judge score despite low hit rate (templated text) |

## Why these documents
Each document type was chosen to stress a different retrieval paradigm, matching the claims made in Section III of the paper:

| Document | Primarily stresses | Because |
|---|---|---|
| `01_company_overview.md` | Vector | General narrative, paraphrasable content |
| `02_hr_policy_manual.md` | Vectorless | Deep, clean header hierarchy (policy → section → subsection) |
| `03_it_security_policy.md` | Vectorless | Same — numbered clauses, exact-lookup style |
| `04_vendor_contract_acme.md` | Vectorless + Graph | Legal structure (traceability) + named parties/dates (entities) |
| `05_project_approval_workflow.md` | Graph | Explicit approval-chain relationships between people/roles |
| `06_quarterly_financial_report_q3.md` | Vector (weak spot) | Contains specific figures, product/cost-center codes that embeddings often miss |
| `07_product_technical_manual.md` | Vectorless | Manual-style structure, exact section lookups |
| `08_supply_chain_network.md` | Graph | Vendor → component → project relationship chains |
| `09_org_chart_and_roles.md` | Graph | Reporting-line relationships referenced by other documents |

## Shared entities (kept consistent across all files — required for multi-hop graph questions)

**People / roles**
- Tom Reyes — CEO
- Sarah Chen — CFO (reports to Tom Reyes)
- Marcus Webb — VP of Engineering (reports to Tom Reyes)
- Priya Raman — Head of Procurement (reports to Sarah Chen)
- David Osei — Director of IT Security (reports to Marcus Webb)
- Elena Petrova — General Counsel (reports to Tom Reyes)
- James Okoro — HR Director (reports to Sarah Chen)
- Aisha Malik — Compliance Officer (reports to Elena Petrova)
- Linda Fischer — VP of Finance Operations (reports to Sarah Chen)

**Projects**
- Project Helios — cloud infrastructure migration (owned by Marcus Webb's Engineering org)
- Project Atlas — ERP system rollout (owned by Linda Fischer's Finance org)

**Vendors**
- Acme Cloud Solutions — cloud compute/storage vendor (supports Project Helios)
- Bright Steel Supply — hardware components vendor (supports Project Helios's on-prem hardware)
- Vertex Logistics — shipping/logistics vendor (supports Project Atlas hardware rollout)


