"""
Enterprise Infrastructure Advisor - Multi-Domain Agentic Workflow
One app for Storage, Server/Compute, Database, and Middleware decisions.

LangGraph agentic workflow (domain-agnostic) × Anthropic Claude ×
native web search × Procurement RAG (TF-IDF) × deterministic TCO.
A supervisor node classifies each question and routes the graph down
one of four paths, so nodes only run when the question actually needs
them - not a fixed pipeline that always does the same work.
Runs on Streamlit Cloud with a single ANTHROPIC_API_KEY.
"""

import os
import json
import operator
import logging
from typing import TypedDict, List, Optional, Dict, Any, Annotated, Literal
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import streamlit as st
from langgraph.graph import StateGraph, END

# Streamlit Cloud exposes secrets via st.secrets, not os.environ - but every
# Config class here reads os.environ at import time (Cloud Run habit, kept
# for local-dev parity with a plain .env). Bridge the two before anything
# downstream imports Config, so both deployment targets work unmodified.
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass  # no secrets.toml present (e.g. local dev using real env vars) - fine

from domains import DOMAINS, DEPLOYMENTS, get_matching_vendors
from advisor_extensions import ProcurementRAG, TCOEngine
from compliance import run_compliance_checks
from arb_report import build_arb_document, build_blueprint_arb_document
from blueprints import BLUEPRINTS, derive_components, analyze_synergy, default_params
from llm_client import Config, get_anthropic_client, MarketIntel, AIAnalyzer
from supervisor import classify_intent, ROUTE_LABELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("infra-advisor")

st.set_page_config(page_title="Infrastructure Advisor - Claude", page_icon="🏛️", layout="wide")

# ==========================================================
# Requirements (domain-generic)
# ==========================================================

@dataclass
class Requirements:
    domain: str
    workload: str
    capacity: int          # in domain units (TB, servers, TB data, instances)
    priority: str
    deployment: str
    availability_target: str

    @property
    def unit(self) -> str:
        return DOMAINS[self.domain]["unit"]

    def metrics(self) -> Dict[str, str]:
        return DOMAINS[self.domain]["workloads"].get(self.workload, {}).get("metrics", {})

    def to_search_prompt(self) -> str:
        year = datetime.now().year
        keywords = DOMAINS[self.domain]["workloads"].get(self.workload, {}).get("keywords", "")
        return (
            f"Summarize the current ({year}) enterprise {self.domain.lower()} market for "
            f"{self.workload} workloads ({self.deployment} deployment; {keywords}). "
            f"Cover leading vendors, recent announcements, and pricing trends. Under 250 words."
        )

# ==========================================================
# LangGraph Workflow (domain-agnostic) — Anthropic-backed, supervised
# ==========================================================

class AgentState(TypedDict):
    requirements: Requirements
    user_query: str
    route_decision: str
    messages: Annotated[List[str], operator.add]
    market_data: Dict[str, Any]
    architecture_analysis: Dict[str, Any]
    vendor_candidates: List[str]
    vendor_recommendations: List[Dict[str, Any]]
    procurement_context: List[Dict[str, Any]]
    tco_estimates: List[Dict[str, Any]]
    compliance_results: List[Dict[str, Any]]
    final_report: Dict[str, Any]
    current_step: str


@st.cache_resource
def create_advisor_graph():
    client = get_anthropic_client()
    intel = MarketIntel(client)
    ai = AIAnalyzer(client)
    rag = ProcurementRAG()

    def classify_query(state: AgentState) -> dict:
        """Supervisor node: decides which downstream nodes actually need to
        run. With no free-text question this always resolves to 'full' -
        existing structured-input behavior is unchanged. Uses a small/cheap
        model since this is classification, not reasoning."""
        req = state["requirements"]
        route, rationale = classify_intent(client, req, state.get("user_query", ""))
        return {"current_step": "routing", "route_decision": route,
                "messages": [f"{ROUTE_LABELS[route]} — {rationale}"]}

    def route_after_classify(state: AgentState) -> Literal["gather_intelligence", "find_vendors"]:
        return "find_vendors" if state["route_decision"] in ("tco_focus", "compliance_focus") \
            else "gather_intelligence"

    def gather_market_intelligence(state: AgentState) -> dict:
        req = state["requirements"]
        market_data = intel.search(req.to_search_prompt(), req)
        return {"current_step": "market_intelligence",
                "messages": [f"✅ Market insights gathered for {req.domain} / {req.workload}"],
                "market_data": market_data}

    def route_after_intel(state: AgentState) -> Literal["analyze_architecture", "generate_report"]:
        return "generate_report" if state["route_decision"] == "market_only" else "analyze_architecture"

    def analyze_architecture(state: AgentState) -> dict:
        req = state["requirements"]
        analysis = ai.analyze_architecture(req, state.get("market_data", {}))
        return {"current_step": "architecture_analysis",
                "architecture_analysis": analysis,
                "messages": ["✅ Architecture analysis complete (Claude)"]}

    def find_vendor_candidates(state: AgentState) -> dict:
        req = state["requirements"]
        matching = get_matching_vendors(req.domain, req.workload, req.deployment, req.capacity)
        return {"current_step": "vendor_matching",
                "vendor_candidates": matching,
                "messages": [f"✅ Found {len(matching)} matching {req.domain.lower()} vendors"]}

    def route_after_vendor_match(state: AgentState) -> Literal["vendors_found", "no_vendors"]:
        return "vendors_found" if state.get("vendor_candidates") else "no_vendors"

    def retrieve_procurement(state: AgentState) -> dict:
        req = state["requirements"]
        candidates = state.get("vendor_candidates", [])
        query = (f"{req.domain} {req.workload} procurement agreements pricing discounts "
                 f"policy for vendors: {', '.join(candidates)}")
        context = rag.retrieve(query, top_k=3, domain=req.domain)
        msg = (f"📄 Retrieved {len(context)} procurement documents (RAG)"
               if context else "📄 No relevant procurement documents found")
        return {"current_step": "procurement_rag", "procurement_context": context, "messages": [msg]}

    def evaluate_vendors(state: AgentState) -> dict:
        req = state["requirements"]
        candidates = state.get("vendor_candidates", [])
        if not candidates:
            return {"current_step": "vendor_evaluation", "vendor_recommendations": [],
                    "tco_estimates": [], "messages": []}
        procurement = state.get("procurement_context", [])
        recs = ai.evaluate_vendors(req, candidates, state.get("market_data", {}), procurement)

        discounts = rag.negotiated_discounts(req.domain)
        vendor_db = DOMAINS[req.domain]["vendors"]
        tco_cfg = DOMAINS[req.domain]["tco"]
        tco_estimates = []
        for rec in recs:
            meta = vendor_db.get(rec.get("name", ""))
            if meta:
                tco_estimates.append(
                    TCOEngine.estimate(rec["name"], meta, req.capacity, discounts, tco_cfg))
        return {"current_step": "vendor_evaluation",
                "vendor_recommendations": recs,
                "tco_estimates": tco_estimates,
                "messages": [f"✅ Evaluated {len(recs)} vendors (Claude) · 💰 TCO computed for {len(tco_estimates)}"]}

    def route_after_evaluate(state: AgentState) -> Literal["compliance_check", "generate_report"]:
        return "generate_report" if state["route_decision"] == "tco_focus" else "compliance_check"

    def explain_no_vendors(state: AgentState) -> dict:
        req = state["requirements"]
        vendor_db = DOMAINS[req.domain]["vendors"]
        workload_vendors = [v for v, d in vendor_db.items() if req.workload in d.get("workloads", [])]
        suggestions = []
        if workload_vendors:
            suggestions.append(
                f"Found {len(workload_vendors)} {req.domain.lower()} vendors supporting {req.workload}, "
                "but they don't match your deployment model or scale requirements")
            suggestions.append(
                f"Consider {'Hybrid' if req.deployment != 'Hybrid' else 'On-Premises or Cloud'} deployment")
        else:
            suggestions.append(f"No vendors in database specialize in {req.workload}")
        cfg = DOMAINS[req.domain]["capacity_slider"]
        if req.capacity <= cfg["min"] * 2:
            suggestions.append(
                f"Scale ({req.capacity} {req.unit}) is small - consider cloud/managed or entry-level options")
        return {"current_step": "no_vendors_found",
                "vendor_recommendations": [{"name": "No Suitable Vendors", "fit_score": 0,
                                            "strengths": [], "considerations": suggestions}],
                "tco_estimates": [],
                "messages": ["⚠️ No suitable vendors found"]}


    def compliance_check(state: AgentState) -> dict:
        """Node: deterministic bank-policy guardrails - no LLM involvement."""
        req = state["requirements"]
        recs = state.get("vendor_recommendations", [])
        if not recs or recs[0].get("fit_score", 0) == 0:
            return {"current_step": "compliance_check", "compliance_results": [], "messages": []}
        # Domain-scoped preferred list + rich agreement status (expiry-aware)
        preferred = rag.preferred_vendors(req.domain)
        agr_status = rag.agreement_status(req.domain)
        results = run_compliance_checks(
            domain=req.domain, workload=req.workload, deployment=req.deployment,
            availability_target=req.availability_target,
            vendor_recommendations=recs,
            vendor_db=DOMAINS[req.domain]["vendors"],
            preferred_vendors=preferred,
            tco_estimates=state.get("tco_estimates", []),
            agreement_status=agr_status,
        )
        warns = sum(1 for r in results if r["overall"] == "warn")
        fails = sum(1 for r in results if r["overall"] == "fail")
        return {"current_step": "compliance_check",
                "compliance_results": results,
                "messages": [f"🛡️ Compliance checks: {len(results)} vendors - {fails} fail, {warns} need review"]}

    def generate_report(state: AgentState) -> dict:
        req = state["requirements"]
        vendors = state.get("vendor_recommendations", [])
        has_real = bool(vendors) and vendors[0].get("fit_score", 0) > 0
        report = {
            "requirements_summary": {
                "domain": req.domain, "workload": req.workload,
                "scale": f"{req.capacity} {req.unit}", "deployment": req.deployment,
                "priority": req.priority, "availability_sla": req.availability_target,
            },
            "architecture": state.get("architecture_analysis", {}),
            "market_sources": state.get("market_data", {}).get("sources", []),
            "procurement_documents_used": [d["id"] for d in state.get("procurement_context", [])],
            "tco_estimates_3yr": state.get("tco_estimates", []),
            "compliance_results": state.get("compliance_results", []),
            "top_vendors": vendors[:3],
            "next_steps": [
                "Engage with top 2-3 vendors for detailed sizing",
                "Request formal quotations with 3-year TCO breakdown",
                "Plan proof-of-concept with representative workload",
                "Validate performance against requirements",
            ] if has_real else [
                "Reassess deployment model requirements",
                "Consider multi-vendor or hybrid approaches",
                "Consult with vendors for custom solutions",
            ],
        }
        return {"current_step": "report_generation", "final_report": report,
                "messages": ["✅ Analysis complete!"]}

    wf = StateGraph(AgentState)
    wf.add_node("classify_query", classify_query)
    wf.add_node("gather_intelligence", gather_market_intelligence)
    wf.add_node("analyze_architecture", analyze_architecture)
    wf.add_node("find_vendors", find_vendor_candidates)
    wf.add_node("retrieve_procurement", retrieve_procurement)
    wf.add_node("evaluate_vendors", evaluate_vendors)
    wf.add_node("explain_no_vendors", explain_no_vendors)
    wf.add_node("compliance_check", compliance_check)
    wf.add_node("generate_report", generate_report)

    wf.set_entry_point("classify_query")

    # Supervisor decides the starting leg: 'full'/'market_only' still open with
    # market intel; 'tco_focus'/'compliance_focus' skip straight to vendor matching.
    wf.add_conditional_edges("classify_query", route_after_classify,
                             {"gather_intelligence": "gather_intelligence", "find_vendors": "find_vendors"})

    # 'market_only' stops after market intel instead of continuing into
    # architecture/vendor/TCO/compliance - it was never asked for.
    wf.add_conditional_edges("gather_intelligence", route_after_intel,
                             {"analyze_architecture": "analyze_architecture", "generate_report": "generate_report"})
    wf.add_edge("analyze_architecture", "find_vendors")

    wf.add_conditional_edges("find_vendors", route_after_vendor_match,
                             {"vendors_found": "retrieve_procurement", "no_vendors": "explain_no_vendors"})
    wf.add_edge("retrieve_procurement", "evaluate_vendors")

    # 'tco_focus' stops after TCO is computed - compliance wasn't asked for.
    wf.add_conditional_edges("evaluate_vendors", route_after_evaluate,
                             {"compliance_check": "compliance_check", "generate_report": "generate_report"})
    wf.add_edge("compliance_check", "generate_report")
    wf.add_edge("explain_no_vendors", "generate_report")
    wf.add_edge("generate_report", END)
    return wf.compile()

# ==========================================================
# UI
# ==========================================================

def render_sidebar():
    """Returns (mode, payload, user_query). user_query is always '' for
    Solution Blueprint mode - the supervisor routing below is scoped to
    Single Domain mode to keep the blueprint's cross-domain correlation
    logic (which needs every component fully analyzed) unaffected."""
    with st.sidebar:
        st.header("📋 Requirements")

        mode = st.radio("Mode", ["Single Domain", "🧩 Solution Blueprint"], horizontal=True,
                        help="Blueprint mode correlates multiple domains with linked sizing")

        if mode == "🧩 Solution Blueprint":
            bp_name = st.selectbox("Use Case", list(BLUEPRINTS.keys()))
            bp = BLUEPRINTS[bp_name]
            st.caption(bp["description"])
            drv = bp["driver"]
            driver_value = st.slider(drv["label"], drv["min"], drv["max"], drv["default"], drv["step"])
            deployment = st.selectbox("Deployment", DEPLOYMENTS)
            priority = st.selectbox("Design Priority", ["Performance", "Cost", "Scalability", "Simplicity"])
            availability = st.selectbox("Availability SLA", ["99.9%", "99.99%", "99.999%"], index=1)

            # Sizing assumptions: transparent AND editable - nothing hardcoded
            params = dict(default_params(bp_name))
            with st.expander("⚙️ Sizing assumptions (editable)", expanded=False):
                st.caption("Deterministic formulas · your inputs. Adjust to your standards; "
                           "the stack re-derives instantly.")
                for key, spec in bp.get("params", {}).items():
                    params[key] = st.number_input(
                        spec["label"], min_value=spec["min"], max_value=spec["max"],
                        value=spec["default"], step=spec["step"],
                        help=spec.get("help", ""), key=f"bp_{bp_name}_{key}")

            comps = derive_components(bp_name, driver_value, params)
            with st.expander("📐 Derived sizing (correlated)", expanded=True):
                for c in comps:
                    unit = DOMAINS[c["domain"]]["unit"]
                    st.write(f"**{c['domain']}** · {c['workload']} → **{c['capacity']} {unit}**")
                    st.caption(c["rationale"])

            st.divider()
            if st.button("🚀 Analyze Full Stack", type="primary", use_container_width=True):
                return "solution", {"blueprint": bp_name, "driver_value": driver_value,
                                    "sizing_params": params,
                                    "components": comps, "deployment": deployment,
                                    "priority": priority, "availability": availability}, ""
            return None, None, ""

        domain = st.selectbox("🏛️ Domain", list(DOMAINS.keys()),
                              help="Which infrastructure decision are you making?")
        cfg = DOMAINS[domain]

        workload = st.selectbox("Workload Type", list(cfg["workloads"].keys()))
        s = cfg["capacity_slider"]
        capacity = st.slider(s["label"], s["min"], s["max"], s["default"], s["step"])
        deployment = st.selectbox("Deployment", DEPLOYMENTS)
        priority = st.selectbox("Design Priority", ["Performance", "Cost", "Scalability", "Simplicity"])
        availability = st.selectbox("Availability SLA", ["99.9%", "99.99%", "99.999%"], index=1)

        user_query = st.text_input(
            "🎯 Ask a specific question (optional)",
            placeholder="e.g. \"just the cost comparison\" or \"is this compliant for on-prem?\"",
            help="Leave blank for the complete analysis. A specific question lets the "
                 "supervisor skip steps it doesn't need - e.g. a cost-only question "
                 "skips market research and compliance.")

        st.divider()
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            return ("single",
                    Requirements(domain, workload, capacity, priority, deployment, availability),
                    user_query)

        with st.expander("ℹ️ About"):
            st.info(f"""
**One advisor, four domains:** {', '.join(DOMAINS.keys())}

**Agentic Stack (Anthropic Claude):**
- 🧭 Supervisor node routes each question down only the path it needs
- 🧠 Claude reasoning · 🔍 Claude native web search
- 📄 Procurement RAG (TF-IDF, local corpus)
- 💰 Deterministic 3-yr TCO engine
- 🔐 Single ANTHROPIC_API_KEY - runs on Streamlit Cloud
            """)
    return None, None, ""


PROGRESS_STEPS = ["routing", "market_intelligence", "architecture_analysis", "vendor_matching",
                  "procurement_rag", "vendor_evaluation", "compliance_check", "report_generation"]
STEP_LABELS = {
    "routing": "🧭 Supervisor Routing",
    "market_intelligence": "🔍 Market Intel", "architecture_analysis": "🏗️ Architecture",
    "vendor_matching": "🔎 Vendor Match", "procurement_rag": "📄 Procurement RAG",
    "vendor_evaluation": "🏢 Evaluation", "compliance_check": "🛡️ Compliance", "no_vendors_found": "⚠️ No Match",
    "report_generation": "📋 Report",
}

def render_workflow_progress(current: str):
    slot = "vendor_evaluation" if current == "no_vendors_found" else current
    if slot in PROGRESS_STEPS:
        idx = PROGRESS_STEPS.index(slot)
        st.progress((idx + 1) / len(PROGRESS_STEPS),
                    text=f"Current: {STEP_LABELS.get(current, 'Processing...')}")



# ==========================================================
# Provenance badges - show WHERE each result comes from
# ==========================================================

PROV = {
    "search":  ("🔍 LIVE WEB",        "#1a56b8", "Claude native web search - real-time, cited"),
    "llm":     ("🤖 LLM (CLAUDE)",    "#b3261e", "Anthropic Claude - AI judgment, human review required"),
    "rag":     ("📄 RAG",             "#137333", "TF-IDF retrieval over internal procurement docs"),
    "rules":   ("⚙️ RULE ENGINE",     "#137333", "Deterministic code - auditable, no LLM"),
    "math":    ("🧮 DETERMINISTIC",   "#137333", "Computed cost model - the LLM never invents a number"),
    "hybrid":  ("🤖+📄 LLM × RAG",    "#7627bb", "Claude ranking grounded by retrieved procurement context"),
}

def provenance(kind: str):
    """Render a small source-of-truth chip under a section header."""
    label, color, tip = PROV[kind]
    st.markdown(
        f"<span style='background:{color}1A;color:{color};border:1px solid {color};"
        f"border-radius:12px;padding:2px 10px;font-size:0.72rem;font-weight:600;'"
        f" title='{tip}'>{label}</span> <span style='color:#5f6368;font-size:0.72rem;'>{tip}</span>",
        unsafe_allow_html=True)


def render_provenance_legend():
    with st.expander("🧭 How to read this analysis - what comes from where", expanded=False):
        st.markdown("""
| Badge | Source | Trust model |
|---|---|---|
| 🧭 **SUPERVISOR** | Claude (small model), routing only | Decides which steps below actually ran |
| 🔍 **LIVE WEB** | Claude native web search | Real-time market data with cited URLs |
| 🤖 **LLM (Claude)** | Anthropic Claude, JSON mode | AI judgment - ranks & explains, never prices |
| 📄 **RAG** | TF-IDF over internal docs | Retrieval with relevance scores - inspectable |
| ⚙️ **RULE ENGINE** | Deterministic Python | Policy & matching as code - fully auditable |
| 🧮 **DETERMINISTIC** | TCO cost model | Transparent formula, negotiated discounts applied |

**Design principle:** AI where judgment helps · code where auditability matters · human sign-off where accountability lives.
""")


def render_results(state: AgentState):
    req = state["requirements"]
    report = state.get("final_report", {})
    vendor_db = DOMAINS[req.domain]["vendors"]

    st.header(f"📊 {req.domain} Analysis Summary")
    if route := state.get("route_decision"):
        st.caption(ROUTE_LABELS.get(route, ""))
    render_provenance_legend()
    cols = st.columns(4)
    cols[0].metric("Scale", f"{req.capacity} {req.unit}")
    metric_items = list(req.metrics().items())
    for i, (k, v) in enumerate(metric_items[:2], start=1):
        cols[i].metric(k, v)
    cols[3].metric("SLA", req.availability_target)

    if market_data := state.get("market_data"):
        with st.expander("🔍 Market Intelligence"):
            provenance("search")
            st.write(market_data.get("answer") or "No market data available")
            if sources := market_data.get("sources"):
                st.markdown("**Sources:**")
                for src in sources:
                    title = src.get("title") or src.get("url") or "source"
                    st.markdown(f"- [{title}]({src.get('url', '#')})")

    st.subheader("🏗️ Architecture Recommendations")
    provenance("llm")
    if arch := state.get("architecture_analysis"):
        if "error" in arch:
            st.warning(f"⚠️ {arch.get('error')}")
        if arch_type := arch.get("architecture_type"):
            st.info(f"**Recommended:** {arch_type}")
        for rec in arch.get("key_recommendations", []):
            st.write(f"• {rec}")
        if notes := arch.get("scalability_notes"):
            st.caption(f"📈 Scalability: {notes}")
        if redundancy := arch.get("redundancy_approach"):
            st.caption(f"🛡️ Redundancy: {redundancy}")

    if procurement := state.get("procurement_context"):
        with st.expander(f"📄 Internal Procurement Context ({len(procurement)} documents retrieved)"):
            provenance("rag")
            for doc in procurement:
                meta = doc.get("meta", {})
                vendor_tag = f" · vendor: {meta['vendor']}" if meta.get("vendor") not in (None, "", "none") else ""
                discount_tag = f" · discount: {meta['discount_pct']}%" if meta.get("discount_pct") else ""
                st.markdown(f"**{doc['id']}** (relevance: {doc.get('score', 0):.2f}{vendor_tag}{discount_tag})")
                st.caption(doc["text"][:300] + ("…" if len(doc["text"]) > 300 else ""))

    if candidates := state.get("vendor_candidates"):
        st.caption(f"💡 {len(candidates)} vendors matched your workload and deployment criteria")
        provenance("rules")

    if tco_list := state.get("tco_estimates"):
        st.subheader("💰 3-Year TCO Comparison")
        provenance("math")
        chart_df = pd.DataFrame({
            "Vendor": [t["vendor"] for t in tco_list],
            "3-Year TCO ($)": [t["total_3yr"] for t in tco_list],
        }).set_index("Vendor")
        st.bar_chart(chart_df)
        tcols = st.columns(min(len(tco_list), 4))
        for col, t in zip(tcols, tco_list[:4]):
            with col:
                delta = f"-{t['negotiated_discount_pct']}% negotiated" if t["has_agreement"] else None
                st.metric(t["vendor"], TCOEngine.fmt(t["total_3yr"]), delta,
                          delta_color="inverse" if delta else "off")
                st.caption(f"{t['model']} · {TCOEngine.fmt(t['range'][0])}–{TCOEngine.fmt(t['range'][1])}")
        st.caption("Model: domain list rate by cost profile (or cloud tier) − negotiated discounts "
                   "from procurement docs + facilities + migration baseline. Rates in domains.py.")

        # --- Sensitivity scenarios (deterministic overlay) ---
        with st.expander("📊 TCO Sensitivity (Base / Optimistic / Conservative)", expanded=False):
            st.caption(TCOEngine.sensitivity_note())
            # Use the lowest-TCO vendor with an agreement as the reference; else top of list
            ref = next((t for t in sorted(tco_list, key=lambda x: x["total_3yr"])
                        if t.get("has_agreement")), tco_list[0])
            scenarios = TCOEngine.sensitivity(ref)
            scols = st.columns(3)
            for col, key in zip(scols, ("optimistic", "base", "conservative")):
                s = scenarios[key]
                with col:
                    st.metric(s["label"], TCOEngine.fmt(s["total_3yr"]))
                    a = s["assumptions"]
                    st.caption(
                        f"Discount {a['discount_pct']}% · "
                        f"Capacity ×{a['capacity_multiplier']} · "
                        f"Facilities ×{a['facilities_multiplier']}"
                    )
            st.caption(f"Reference vendor: **{ref['vendor']}** (lowest modelled TCO with agreement, or first).")

        # --- Realized vs model overlay (from PO history corpus) ---
        try:
            _rag = ProcurementRAG()
            realized = _rag.realized_costs()
            if realized.get("po_count", 0) > 0:
                with st.expander("📉 Model vs Realized Unit Cost (from PO history)", expanded=False):
                    st.caption(
                        f"Parsed {realized['po_count']} POs from "
                        f"{', '.join(realized.get('source_docs') or [])}. "
                        "Comparison is indicative — different generations and configs apply."
                    )
                    unit = req.unit if hasattr(req, "unit") else "TB"
                    rows = []
                    for t in tco_list:
                        cmp_ = TCOEngine.compare_to_realized(t, realized, unit_label=unit)
                        if cmp_:
                            rows.append({
                                "Vendor": t["vendor"],
                                "Model / unit": TCOEngine.fmt(cmp_["model_per_unit"]),
                                "Realized / unit": TCOEngine.fmt(cmp_["realized_per_unit"]),
                                "Δ %": f"{cmp_['delta_pct']:+.1f}%",
                                "Signal": cmp_["interpretation"],
                                "Source": cmp_["source"],
                            })
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    else:
                        st.info("No overlapping vendor realized-cost signal for this domain.")
        except Exception:
            pass  # non-fatal — corpus may be absent in some test contexts


    if compliance := state.get("compliance_results"):
        st.subheader("🛡️ Compliance Guardrails")
        provenance("rules")
        badge = {"pass": "✅ PASS", "warn": "🟡 REVIEW", "fail": "🔴 FAIL"}
        for block in compliance:
            with st.expander(f"{badge[block['overall']]} - {block['vendor']}",
                             expanded=(block["overall"] != "pass")):
                for c in block["checks"]:
                    icon = {"pass": "✅", "warn": "🟡", "fail": "🔴"}[c["status"]]
                    st.write(f"{icon} **{c['name']}** - {c['detail']}")

    st.subheader("🏆 Vendor Recommendations")
    provenance("hybrid")
    if vendors := state.get("vendor_recommendations"):
        for i, vendor in enumerate(vendors[:4], 1):
            if not vendor:
                continue
            score = vendor.get("fit_score", 0)
            name = vendor.get("name", "Unknown")
            if "error" in vendor or name == "Error":
                with st.expander("⚠️ Evaluation Error", expanded=True):
                    st.error(f"Error: {vendor.get('error', 'Unknown error')}")
                    for c in vendor.get("considerations", []):
                        st.write(f"• {c}")
                continue
            if score == 0:
                with st.expander(f"⚠️ {name}", expanded=True):
                    for c in vendor.get("considerations", []):
                        st.warning(f"• {c}")
            else:
                meta = vendor_db.get(name, {})
                comp = next((c for c in state.get("compliance_results", []) if c["vendor"] == name), None)
                comp_badge = {"pass": " | 🛡️✅", "warn": " | 🛡️🟡", "fail": " | 🛡️🔴"}.get(comp["overall"], "") if comp else ""
                with st.expander(f"{i}. {name} - Score: {score}/10 | Cost: {meta.get('cost_profile', 'N/A').title()}{comp_badge}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Strengths:**")
                        for s in vendor.get("strengths", []):
                            st.write(f"✓ {s}")
                        if sweet := meta.get("sweet_spot"):
                            st.caption(f"📏 Sweet Spot: {sweet}")
                    with c2:
                        st.markdown("**Considerations:**")
                        for c in vendor.get("considerations", []):
                            st.write(f"• {c}")
                    tco = next((t for t in state.get("tco_estimates", []) if t["vendor"] == name), None)
                    if tco:
                        agreement = " (existing agreement applied)" if tco["has_agreement"] else ""
                        st.info(f"💰 **3-Yr TCO:** {TCOEngine.fmt(tco['range'][0])}–{TCOEngine.fmt(tco['range'][1])}{agreement}")
                    if services := meta.get("services"):
                        st.caption(" · ".join(f"{k.title()}: {v}" for k, v in services.items()))
    else:
        st.info("No vendor recommendations available")

    if next_steps := report.get("next_steps"):
        st.subheader("🚀 Next Steps")
        for i, step in enumerate(next_steps, 1):
            st.write(f"{i}. {step}")

    st.divider()
    _, col1, col2 = st.columns([3, 1.4, 1])
    with col1:
        if report and state.get("vendor_recommendations"):
            try:
                arb_bytes = build_arb_document(state, TCOEngine.fmt)
                st.download_button(
                    "📄 ARB Decision Record (.docx)", arb_bytes,
                    f"ARB_{req.domain.split()[0]}_{datetime.now():%Y%m%d_%H%M%S}.docx",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    type="primary")
            except Exception as e:
                logger.error(f"ARB generation failed: {e}")
                st.caption("ARB document generation unavailable")
    with col2:
        if report:
            st.download_button("📥 JSON", json.dumps(report, indent=2, default=str),
                               f"{req.domain.split()[0].lower()}_{datetime.now():%Y%m%d_%H%M%S}.json",
                               "application/json")

# ==========================================================
# Main
# ==========================================================


def _blank_state(req: Requirements) -> dict:
    return {"requirements": req, "user_query": "", "route_decision": "", "messages": [],
            "market_data": {}, "architecture_analysis": {}, "vendor_candidates": [],
            "vendor_recommendations": [], "procurement_context": [],
            "tco_estimates": [], "compliance_results": [], "final_report": {},
            "current_step": "initializing"}


def run_solution(payload: dict):
    """Run the (unchanged) LangGraph workflow once per correlated component."""
    workflow = create_advisor_graph()
    results = []
    progress = st.progress(0.0, text="Starting stack analysis...")
    comps = payload["components"]
    for i, c in enumerate(comps):
        req = Requirements(c["domain"], c["workload"], c["capacity"],
                           payload["priority"], payload["deployment"], payload["availability"])
        progress.progress(i / len(comps), text=f"Analyzing {c['domain']} · {c['workload']}...")
        final = dict(_blank_state(req))
        for update in workflow.stream(_blank_state(req)):
            if isinstance(update, dict):
                delta = list(update.values())[0]
                for k, v in delta.items():
                    if k == "messages":
                        final["messages"] = final.get("messages", []) + v
                    else:
                        final[k] = v
        results.append({"domain": c["domain"], "workload": c["workload"],
                        "capacity": c["capacity"], "unit": DOMAINS[c["domain"]]["unit"],
                        "rationale": c["rationale"], "state": final})
    progress.progress(1.0, text="Stack analysis complete")
    return results


def render_solution_results(payload: dict, results: list):
    st.header(f"🧩 {payload['blueprint']} - Full-Stack Analysis")
    st.caption(f"Driver: {payload['driver_value']} · Deployment: {payload['deployment']} · "
               f"SLA: {payload['availability']} · Components sized by correlated formulas")

    synergy = analyze_synergy(results)

    # Stack summary metrics
    cols = st.columns(len(results) + 1)
    total = synergy.get("estimated_stack_tco_3yr", 0)
    cols[0].metric("Stack 3-Yr TCO", TCOEngine.fmt(total) if total else "N/A")
    for col, comp in zip(cols[1:], results):
        top = next((r for r in comp["state"].get("vendor_recommendations", [])
                    if r.get("fit_score", 0) > 0), None)
        col.metric(comp["domain"].split(" /")[0], top["name"] if top else "-",
                   f"{comp['capacity']} {comp['unit']}", delta_color="off")

    # Cross-domain synergy
    st.subheader("🔗 Cross-Domain Vendor Correlation")
    provenance("rules")
    if synergy["bundle_opportunities"]:
        for b in synergy["bundle_opportunities"]:
            st.success(f"💼 {b}")
        for note in synergy["concentration_notes"]:
            st.warning(f"⚖️ {note}")
    else:
        st.info("No single vendor spans multiple components - a best-of-breed stack. "
                "Consider integration effort in delivery planning.")

    # Combined TCO chart (top vendor per component)
    rows = []
    for comp in results:
        state = comp["state"]
        top = next((r for r in state.get("vendor_recommendations", []) if r.get("fit_score", 0) > 0), None)
        if top:
            tco = next((t for t in state.get("tco_estimates", []) if t["vendor"] == top["name"]), None)
            if tco:
                rows.append({"Component": f"{comp['domain'].split(' /')[0]}\n{top['name']}",
                             "3-Year TCO ($)": tco["total_3yr"]})
    if rows:
        st.subheader("💰 Stack TCO by Component (top-ranked vendor each)")
        df = pd.DataFrame(rows).set_index("Component")
        st.bar_chart(df)

    # Per-component detail
    st.subheader("📦 Component Analyses")
    for comp in results:
        top = next((r for r in comp["state"].get("vendor_recommendations", [])
                    if r.get("fit_score", 0) > 0), None)
        header = (f"{comp['domain']} · {comp['workload']} · {comp['capacity']} {comp['unit']}"
                  + (f" → {top['name']}" if top else " → no match"))
        with st.expander(header):
            st.caption(f"Sizing rationale: {comp['rationale']}")
            render_results(comp["state"])

    # Export
    st.divider()
    export = {
        "blueprint": payload["blueprint"],
        "driver_value": payload["driver_value"],
        "sizing_assumptions": payload.get("sizing_params", {}),
        "deployment": payload["deployment"],
        "availability": payload["availability"],
        "synergy": synergy,
        "components": [{"domain": c["domain"], "workload": c["workload"],
                        "capacity": c["capacity"], "unit": c["unit"],
                        "rationale": c["rationale"],
                        "report": c["state"].get("final_report", {})} for c in results],
    }
    _, col1, col2 = st.columns([3, 1.4, 1])
    with col1:
        try:
            bp_spec = BLUEPRINTS.get(payload["blueprint"], {})
            param_labels = {k: v.get("label", k) for k, v in bp_spec.get("params", {}).items()}
            arb_bytes = build_blueprint_arb_document(
                blueprint_name=payload["blueprint"],
                description=bp_spec.get("description", ""),
                driver_label=bp_spec.get("driver", {}).get("label", "Driver"),
                driver_value=payload["driver_value"],
                params=payload.get("sizing_params", {}),
                param_labels=param_labels,
                component_results=results,
                synergy=synergy,
                fmt_money=TCOEngine.fmt,
            )
            st.download_button(
                "📄 ARB Solution Record (.docx)", arb_bytes,
                f"ARB_Blueprint_{payload['blueprint'].split()[0]}_{datetime.now():%Y%m%d_%H%M%S}.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary")
        except Exception as e:
            logger.error(f"Solution ARB generation failed: {e}")
            st.caption("ARB document generation unavailable")
    with col2:
        st.download_button("📥 Stack Report (JSON)", json.dumps(export, indent=2, default=str),
                           f"stack_{datetime.now():%Y%m%d_%H%M%S}.json", "application/json")


def main():
    st.title("🏛️ Enterprise Infrastructure Advisor")
    st.caption("One agentic advisor for Storage · Server · Database · Middleware — "
               "LangGraph supervisor × Claude × Procurement RAG × deterministic TCO")

    if Config.is_demo_mode():
        st.info(
            "📦 **Demo / Offline mode** — no API key detected. "
            "The full workflow (supervisor routing, architecture, vendor ranking, "
            "RAG, TCO, compliance, ARB export) runs with high-quality local fixtures. "
            "Set `ANTHROPIC_API_KEY` to switch to live Claude + web search.",
            icon="📦",
        )

    if not Config.is_demo_mode() and not Config.validate():
        st.stop()

    if "agent_state" not in st.session_state:
        st.session_state.agent_state = None
    if "solution_results" not in st.session_state:
        st.session_state.solution_results = None

    mode, payload, user_query = render_sidebar()

    if mode == "solution" and payload:
        st.session_state.agent_state = None
        try:
            with st.spinner("Running correlated multi-domain analysis on Claude..."):
                st.session_state.solution_results = {
                    "payload": payload, "results": run_solution(payload)}
            st.success("✅ Full-stack analysis completed!")
        except Exception as e:
            logger.exception("Solution analysis failed")
            st.error(f"❌ Stack analysis failed: {e}")
            st.stop()

    requirements = payload if mode == "single" else None

    if requirements:
        st.session_state.solution_results = None
        initial_state: AgentState = {
            **_blank_state(requirements), "user_query": user_query,
        }
        st.subheader("🔗 Agentic Workflow")
        progress_placeholder = st.empty()
        try:
            workflow = create_advisor_graph()
            with st.spinner(f"Analyzing {requirements.domain} with Claude..."):
                accumulated: Dict[str, Any] = dict(initial_state)
                for state_update in workflow.stream(initial_state):
                    if isinstance(state_update, dict):
                        node_delta = list(state_update.values())[0]
                        for k, v in node_delta.items():
                            if k == "messages":
                                accumulated["messages"] = accumulated.get("messages", []) + v
                            else:
                                accumulated[k] = v
                        with progress_placeholder.container():
                            render_workflow_progress(accumulated.get("current_step", ""))
                st.session_state.agent_state = accumulated
                st.success("✅ Workflow completed!")
        except Exception as e:
            logger.exception("Workflow failed")
            st.error(f"❌ Workflow failed: {e}")
            st.stop()

    if st.session_state.solution_results:
        st.divider()
        render_solution_results(st.session_state.solution_results["payload"],
                                st.session_state.solution_results["results"])
    elif st.session_state.agent_state:
        st.divider()
        render_results(st.session_state.agent_state)
    else:
        domain_list = "\n".join(
            f"- **{name}** - {len(cfg['vendors'])} vendors, {len(cfg['workloads'])} workload profiles"
            for name, cfg in DOMAINS.items())
        st.markdown(f"""
### One App for Every Infrastructure Decision

{domain_list}

**A supervisor routes every question, not a fixed pipeline:**
```
                                    ┌─ "market_only"  ──────────────────────────────┐
Supervisor (Claude) ──┬─ "full" ────► Market Intel → Architecture ─┐                │
                       │                                            ▼                ▼
                       └─ "tco_focus" / "compliance_focus" ──► Vendor Match ─┬─→ Procurement RAG → Evaluation
                                                                              └─→ No Vendors → Guidance
                                                                                        │
                                                            "tco_focus" skips ──────────┼──→ Compliance
                                                                                        ▼
                                                                                     Report
```

Adding a domain (Network, Backup Software, …) is a **data-only change** in `domains.py` -
the workflow, RAG, and TCO engine are domain-agnostic. Ask a specific question in the
sidebar to see the supervisor take a shorter path.

Pick a domain in the sidebar to begin.
        """)


if __name__ == "__main__":
    main()
