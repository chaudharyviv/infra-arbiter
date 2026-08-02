"""
Supervisor / router node.

Without this, the workflow always runs every node regardless of what was
actually asked - a fixed pipeline that happens to use an LLM, not an agent.
This node lets a free-text question steer which nodes actually execute:

- "full"              → run everything (default when no query is given -
                         existing behaviour is preserved exactly)
- "market_only"        → the user just wants current market context; skip
                         architecture, vendor matching, RAG, TCO, compliance
- "tco_focus"          → the user wants cost comparison; skip market intel
                         and architecture narrative, skip compliance
- "compliance_focus"   → the user wants a governance/compliance read; skip
                         market intel and architecture narrative, but still
                         run vendor matching + evaluation (compliance checks
                         need a vendor to evaluate) and compliance_check

Routing itself uses a small/cheap model - it's a classification task, not a
reasoning task, so there's no reason to spend the main model's budget on it.
"""

import os
import json
import re
import logging
from typing import Literal, Tuple

import anthropic

import streamlit as st

logger = logging.getLogger("infra-advisor.supervisor")

Route = Literal["full", "market_only", "tco_focus", "compliance_focus"]

VALID_ROUTES = {"full", "market_only", "tco_focus", "compliance_focus"}


def get_router_model() -> str:
    if "ANTHROPIC_ROUTER_MODEL" in os.environ and os.environ["ANTHROPIC_ROUTER_MODEL"]:
        return os.environ["ANTHROPIC_ROUTER_MODEL"]
    try:
        return str(st.secrets.get("ANTHROPIC_ROUTER_MODEL", "claude-haiku-4-5-20251001"))
    except Exception:
        return "claude-haiku-4-5-20251001"


def classify_intent(client: anthropic.Anthropic, req, user_query: str) -> Tuple[Route, str]:
    """Returns (route, rationale). Falls back to 'full' on empty query or any
    classification failure - the safe default is to run everything, never to
    silently under-run the analysis."""
    if not user_query or not user_query.strip():
        return "full", "No specific question asked - running the complete analysis."

    prompt = f"""A user is using an enterprise infrastructure advisory tool for a
{req.domain} decision ({req.workload}, {req.capacity} {req.unit}). They asked:

"{user_query.strip()}"

Classify what they actually need into exactly one of:
- "full": a complete recommendation (architecture + vendors + cost + compliance)
- "market_only": just current market/vendor landscape context, no recommendation needed
- "tco_focus": primarily a cost/TCO comparison question
- "compliance_focus": primarily a governance/compliance/policy question

Return ONLY JSON: {{"route": "<one of the four>", "rationale": "<one short sentence>"}}"""

    model = get_router_model()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=200,
            system="You are a routing classifier. Return only valid JSON, no other text.",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL)
        parsed = json.loads(raw)
        route = parsed.get("route", "full")
        rationale = parsed.get("rationale", "")
        if route not in VALID_ROUTES:
            logger.warning(f"Router returned invalid route '{route}', falling back to 'full'")
            return "full", "Could not confidently classify the question - running the complete analysis."
        return route, rationale
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return "full", "Routing check failed - running the complete analysis (safe default)."


ROUTE_LABELS = {
    "full": "🧭 Full analysis (architecture + vendors + cost + compliance)",
    "market_only": "🧭 Market-context only - skipping vendor match, TCO, and compliance",
    "tco_focus": "🧭 Cost-focused - skipping market narrative and compliance",
    "compliance_focus": "🧭 Compliance-focused - skipping market narrative, still evaluating vendors",
}
