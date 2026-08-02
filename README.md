# Enterprise Infrastructure Advisor

**One agentic advisor for Storage · Server/Compute · Database · Middleware decisions.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-purple.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent, multi-domain infrastructure advisory platform that combines:

| Capability | Implementation |
|---|---|
| Agentic orchestration | LangGraph with a **supervisor router** that short-circuits unused nodes |
| Reasoning & ranking | Anthropic Claude (architecture narrative + vendor fit) |
| Live market context | Claude native web search (cited sources) |
| Internal knowledge | **Procurement RAG** (TF-IDF over local policy & agreement corpus) |
| Cost modelling | **Deterministic 3-year TCO engine** - never invented by the LLM |
| Cost realism | **Sensitivity scenarios** + **model-vs-realized** overlay from PO history |
| Governance | **Policy-as-code** compliance (SLA, residency, onboarding, concentration, agreement expiry) |
| Cross-domain programmes | **Six Solution Blueprints** with editable, transparent sizing formulas |
| Decision artefacts | One-click **Architecture Review Board (.docx)** decision records |

> **Design principle:** AI where judgment helps · code where auditability matters · human sign-off where accountability lives.

---

## Why this exists

Enterprise infrastructure decisions (storage arrays, database platforms, GPU clusters, messaging fabrics) sit at the intersection of technical fit, commercial agreements, regulatory policy and multi-year cost. Most tools either:

- treat the problem as pure generative text (unreliable numbers, opaque policy), or
- are rigid spreadsheet/CMDB workflows that ignore market context and architectural nuance.

This advisor deliberately splits the work:

1. **Claude** reasons about architecture patterns and ranks vendors against workload + procurement context.
2. **Deterministic Python** computes TCO, applies negotiated discounts (expiry-aware), overlays historical realized unit costs, and evaluates compliance rules from a configurable policy pack.
3. **A formal ARB document** is generated for human review and sign-off.

The result is usable both as a portfolio demonstration of production-grade agent design and as a practical decision-support tool inside an architecture practice.

---

## Quick start (fully offline)

```bash
git clone <this-repo>
cd enterprise-infra-advisor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# No API key required - demo mode activates automatically
streamlit run infra_advisor.py
```

The UI, LangGraph workflow, RAG retrieval, TCO calculations (including sensitivity and realized-cost comparison), compliance checks and ARB document generation all run end-to-end with realistic canned responses when `ANTHROPIC_API_KEY` is absent (or `DEMO_MODE=1` is set).

### Optional live mode

```bash
# .streamlit/secrets.toml  or export
ANTHROPIC_API_KEY=sk-ant-...
# optional overrides
# ANTHROPIC_MODEL=claude-sonnet-4-...
# ANTHROPIC_ROUTER_MODEL=claude-haiku-...
streamlit run infra_advisor.py
```

---

## Architecture

```
                         ┌─ market_only ──────────────────────────────┐
Supervisor (router) ──┬─ full ────► Market Intel → Architecture ─┐    │
                      │                                           ▼    ▼
                      └─ tco / compliance focus ──► Vendor Match ─┬─→ Procurement RAG
                                                                  │     ↓
                                                                  │  Evaluation + TCO
                                                                  │  (+ sensitivity, realized)
                                                                  │     ↓
                                                                  └─→ Compliance (policy pack)
                                                                        ↓
                                                                     Report + ARB .docx
```

| Module | Role |
|---|---|
| `domains.py` | Domain registry - workloads, vendors, TCO rates, capacity sliders. Adding Network or Backup is a data-only change. |
| `supervisor.py` | Intent router. Classifies free-text questions and prunes the graph. Keyword fallback keeps the path visible in demo mode. |
| `advisor_extensions.py` | **ProcurementRAG** (TF-IDF, domain-scoped, expiry-aware discounts, realized-cost extraction) + **TCOEngine** (estimate, sensitivity, model-vs-realized). |
| `compliance.py` | Policy-as-code guardrails via a swapable **PolicyPack** (Tier-1 SLA, residency, onboarding + agreement freshness, concentration). |
| `blueprints.py` | Six cross-domain programmes; deterministic sizing from one driver; synergy / concentration analysis. |
| `arb_report.py` | Single-domain and consolidated solution-level Word decision records. |
| `llm_client.py` | Claude client + high-quality offline fixtures for portfolio / CI. |

---

## Procurement corpus

Eight realistic (anonymised) documents ship under `data/procurement/`:

| Document | Domain | Effect |
|---|---|---|
| `dell_framework_agreement.md` | Storage | 18% discount, preferred vendor, valid until 2027-12-31 |
| `dell_server_addendum.md` | Server / Compute | 15% discount, valid until 2026-12-31 |
| `ela_netapp_2025.md` | Storage | 22% ELA discount |
| `cloud_commit_gcp.md` | Storage | 25% committed-use discount |
| `oracle_ula_2024.md` | Database | 30% ULA |
| `redhat_enterprise_agreement.md` | Middleware | 20% EA |
| `storage_standards_policy.md` | Storage (policy) | Preferred-vendor & residency rules |
| `po_history_2024_2025.md` | Storage (history) | Realised $/TB benchmarks for model comparison |

Frontmatter supplies structured metadata (`vendor`, `discount_pct`, `domain`, `valid_until`, `doc_type`) consumed by:

- domain-scoped RAG retrieval
- the TCO discount engine (**expired agreements are excluded by default**)
- compliance onboarding checks (expiring-soon → REVIEW)
- realized-cost extraction from PO history lines

---

## Solution Blueprints

Six shipped programmes demonstrate cross-domain correlation. Sizing assumptions are first-class, named parameters editable in the sidebar - changing them instantly re-derives the entire stack.

| Blueprint | Driver | Domains | Narrative |
|---|---|---|---|
| **AI/ML Training Platform** | GPU servers | Server · Storage · Database | GPU-correlated dataset + metadata sizing |
| **Core Banking Modernization** | Primary DB size (TB) | Database · Server · Storage · Middleware | Full Tier-1 stack from ledger size |
| **Enterprise Data Lake & Analytics** | Raw data volume (TB) | Storage · Database · Middleware | Lake + warehouse + ingestion |
| **Payments / Real-Time Transaction Platform** | Peak TPS (thousands) | **Middleware** · Database · Storage | Messaging-primary, strict Tier-1 latency path |
| **Hybrid Cloud Landing Zone / VDI Platform** | Concurrent desktops | Server · Storage | Hybrid density, boot-storm factor, concentration |
| **Disaster Recovery / Secondary Site** | Primary footprint (TB) | Storage · Server · Database | Asymmetric DR cost (pilot-light / warm fractions) |

After each stack run the system:

- aggregates top-vendor 3-year TCO
- detects multi-domain vendor overlap (bundle opportunities)
- flags concentration when a single family holds a large spend share
- emits a consolidated **ARB Solution Decision Record** (.docx)

---

## Deterministic cost & governance depth

### TCO engine

- Formula: domain list rate (or cloud tier) × capacity × (1 − negotiated discount) + facilities + migration baseline, with a ±15% sizing band.
- **Sensitivity**: Optimistic / Base / Conservative overlays (discount depth, capacity growth, facilities burden) - pure arithmetic, no LLM.
- **Model vs realized**: when PO history exists, unit cost from the model is compared to historical $/TB (vendor-specific or portfolio average) and surfaced in the UI.

### Compliance (policy-as-code)

Checks per recommended vendor:

1. Tier-1 / production SLA floor  
2. Data residency (in-region + CMEK note for cloud/hybrid)  
3. Vendor onboarding - preferred if active agreement; **expiring-soon agreements raise REVIEW**  
4. Concentration risk against a configurable spend-share threshold  

Rules live in a frozen **`PolicyPack`** dataclass (default: “Bank India / RBI”). Swap the pack, not the check logic.

---

## Testing the deterministic core

```bash
pytest tests/ -v
```

**27 offline tests** cover:

- corpus loading & frontmatter parsing  
- domain-scoped discounts and agreement expiry behaviour  
- realized-cost extraction from PO history  
- TF-IDF retrieval scores  
- TCO formula, uncertainty band, sensitivity scenarios, model-vs-realized  
- all six blueprint sizing formulas (including overrides)  
- vendor matching filters  
- compliance matrix (Tier-1 fail, preferred pass, expiry warn, custom policy pack)

No network or API key required.

---

## Project layout

```
enterprise-infra-advisor/
├── infra_advisor.py          # Streamlit entrypoint + LangGraph workflow
├── domains.py                # Domain registry (vendors, workloads, TCO rates)
├── advisor_extensions.py     # ProcurementRAG + TCOEngine (+ sensitivity, realized)
├── llm_client.py             # Claude client + Demo/Offline fixtures
├── supervisor.py             # Intent router
├── compliance.py             # PolicyPack + guardrail checks
├── blueprints.py             # Six cross-domain programmes & synergy analysis
├── arb_report.py             # ARB Decision Record (.docx) generator
├── data/procurement/         # Local agreement, policy & PO-history corpus
├── tests/                    # Deterministic unit tests (27 cases)
├── docs/                     # Architecture notes
├── scripts/                  # Demo / test helpers
├── .streamlit/               # Theme + secrets example
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Design decisions worth noting

| Decision | Rationale |
|---|---|
| TF-IDF instead of embeddings | Corpus is small and well-structured; deterministic, zero cost, zero provisioning. Interface is already swap-ready for a vector store. |
| Supervisor router | Most “agent” demos run a fixed pipeline. Routing only the nodes that are needed reduces latency, cost and cognitive load. |
| TCO never leaves Python | Pricing is a business decision. The LLM ranks and explains; the engine computes - including sensitivity and realized overlays. |
| Agreement expiry is first-class | Discounts and preferred-vendor status degrade after `valid_until`. Compliance surfaces expiring agreements as REVIEW. |
| Compliance as a PolicyPack | Bank policy must be auditable and swappable (region / organisation) without rewriting check code. |
| Fixed blueprints, not a free-form builder | Correlated sizing from a named driver is the product story. Six realistic programmes beat a generic multi-select form for portfolio clarity and demo reliability. |
| Demo mode first-class | Portfolio and CI exercise the full surface without an API key. Live mode is a drop-in upgrade. |
| Domain registry pattern | New infrastructure domains are data, not code. |

---

## Suggested demo paths

| Scenario | What to notice |
|---|---|
| Database · OLTP / Core Banking · 20 TB · On-Prem · 99.999% | Oracle ULA 30% surfaces via RAG; TCO vs PostgreSQL economics; Tier-1 SLA path |
| Storage · Backup · 300 TB · On-Prem | Dell 18% framework; model-vs-realized $/TB from PO history |
| Blueprint: Payments / Real-Time · 50K TPS | Messaging-primary stack; Tier-1 pressure across domains |
| Blueprint: AI/ML Training · 8 GPU servers | Correlated storage + metadata; possible multi-domain vendor leverage |
| Blueprint: DR Secondary Site · 200 TB primary | Asymmetric compute (40% default) vs full storage replica |
| Free-text: “just the cost comparison” | Supervisor routes `tco_focus` - skips market narrative and compliance |

---

## Roadmap (illustrative)

- Persist analyses (scenario comparison, history)
- Richer TCO drivers (power density, licence uplifts, commitment tiers)
- Real document ingestion path (PDF → chunk → vector) behind the same RAG interface
- Additional regional policy packs
- Side-by-side blueprint scenario comparison in the UI
- Optional “custom stack” mode from approved domain Lego blocks (still deterministic)

---

## License

MIT - see [LICENSE](LICENSE).

---

*Built as a portfolio demonstration of production-minded agentic systems: supervisor routing, hybrid retrieval, deterministic financial & policy engines, agreement-aware commercial logic, cross-domain programme sizing, and formal decision artefacts.*
