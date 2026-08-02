# Architecture Notes

## Control flow

```
User input (sidebar)
        │
        ▼
┌───────────────────┐
│  classify_query   │  Supervisor - route = full | market_only | tco_focus | compliance_focus
└─────────┬─────────┘
          │
    ┌─────┴──────┐
    ▼            ▼
gather_intel   find_vendors          (market_only stops after intel)
    │            │
    ▼            ▼
analyze_arch   retrieve_procurement
    │            │
    └─────► evaluate_vendors ──► compliance_check (skipped on tco_focus)
                   │
                   ▼
            generate_report → END
```

## AI vs Deterministic boundary

| Concern | Owner | Why |
|---|---|---|
| Market narrative + citations | Claude + web_search | Needs current external knowledge |
| Architecture pattern & recommendations | Claude | Judgment-heavy, domain reasoning |
| Vendor ranking & narrative | Claude (grounded by RAG) | Fit is multi-factor; agreements influence score |
| Candidate pool | Rule engine (`get_matching_vendors`) | Auditable filters |
| Negotiated discounts | Frontmatter parser | Exact commercial terms |
| 3-year TCO | `TCOEngine` | Business formula, never LLM |
| Compliance status | `run_compliance_checks` | Policy-as-code |
| Blueprint sizing & synergy | Pure functions | Correlated math, reproducible |
| Final decision record | Human (ARB .docx) | Accountability |

## Extensibility points

1. **New domain** - add an entry to `DOMAINS` in `domains.py`. No graph changes.
2. **New blueprint** - add to `BLUEPRINTS` with driver, params, and `Component` list.
3. **Richer RAG** - replace `ProcurementRAG` internals; keep `retrieve()` / `negotiated_discounts()` signatures.
4. **Live TCO drivers** - extend `tco_cfg` and the estimate formula; UI already surfaces methodology.
5. **Policy packs** - parameterise `compliance.py` rules by organisation / region.

## Demo mode contract

When `Config.is_demo_mode()` is true:

- `get_anthropic_client()` returns `None`
- `MarketIntel.search` returns domain-appropriate canned narrative + sources
- `AIAnalyzer` returns realistic architecture and ranked-vendor fixtures
- `classify_intent` uses transparent keyword routing
- All deterministic nodes (RAG, TCO, compliance, blueprints, ARB) run unchanged

This guarantees the entire product surface is exercisable offline for portfolio demos and CI.
