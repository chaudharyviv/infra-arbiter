"""
Anthropic Claude client — replaces the Vertex AI Gemini client.

Two responsibilities, same interface contract as before so infra_advisor.py's
graph nodes don't change shape:
  - MarketIntel: uses Claude's native web_search server tool for grounded,
    cited market context (replacement for Gemini + Google Search grounding).
  - AIAnalyzer: JSON-mode architecture/vendor analysis (replacement for
    Gemini JSON mode + thinking_budget=0).

Auth: a single ANTHROPIC_API_KEY. No service account, no GCP project.
"""

import os
import re
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

import streamlit as st
import tomllib
import anthropic

logger = logging.getLogger("infra-advisor.llm")


class _ConfigMeta(type):
    @property
    def API_KEY(cls) -> str:
        return cls.get_api_key()

    @property
    def MODEL(cls) -> str:
        return cls.get_model()

    @property
    def MAX_TOKENS(cls) -> int:
        return cls.get_max_tokens()


class Config(metaclass=_ConfigMeta):
    @classmethod
    def get_api_key(cls) -> str:
        # First, check the environment (set by the bridge code or manually)
        env_key = os.getenv("ANTHROPIC_API_KEY", "")
        if env_key:
            env_key = env_key.strip()
            logger.debug("ANTHROPIC_API_KEY loaded from environment – length %d", len(env_key))
            return env_key
        # Fallback to Streamlit secrets (used in local development / Streamlit Cloud)
        try:
            secret_key = str(st.secrets.get("ANTHROPIC_API_KEY", ""))
            if secret_key:
                return secret_key.strip()
        except Exception:
            pass
        # Final fallback to local .streamlit/secrets.toml
        try:
            with open(".streamlit/secrets.toml", "rb") as f:
                data = tomllib.load(f)
                secret_key = str(data.get("ANTHROPIC_API_KEY", ""))
                return secret_key.strip()
        except Exception as e:
            logger.debug("Could not load ANTHROPIC_API_KEY from secrets.toml: %s", e)
            return ""

    @classmethod
    def get_model(cls) -> str:
        if "ANTHROPIC_MODEL" in os.environ and os.environ["ANTHROPIC_MODEL"]:
            return os.environ["ANTHROPIC_MODEL"]
        try:
            return str(st.secrets.get("ANTHROPIC_MODEL", "claude-sonnet-5"))
        except Exception:
            return "claude-sonnet-5"

    @classmethod
    def get_max_tokens(cls) -> int:
        val = os.environ.get("MAX_OUTPUT_TOKENS")
        if not val:
            try:
                val = st.secrets.get("MAX_OUTPUT_TOKENS", "")
            except Exception:
                pass
        try:
            return int(val)
        except (ValueError, TypeError):
            return 4096

    @classmethod
    def validate(cls) -> bool:
        if not cls.get_api_key():
            st.error("❌ ANTHROPIC_API_KEY is not set. Add it under Streamlit Cloud → "
                      "Settings → Secrets (or a local .streamlit/secrets.toml for dev).")
            return False
        return True

    @classmethod
    def is_demo_mode(cls) -> bool:
        """No API key configured -> the whole app runs offline on fixtures
        instead of live Claude calls. See ARCHITECTURE.md's Demo mode contract."""
        return not cls.get_api_key()


@st.cache_resource
def get_anthropic_client() -> Optional[anthropic.Anthropic]:
    if Config.is_demo_mode():
        return None
    return anthropic.Anthropic(api_key=Config.get_api_key())


# ==========================================================
# Market Intelligence — Claude + native web_search tool
# ==========================================================

class MarketIntel:
    def __init__(self, client: Optional[anthropic.Anthropic]):
        self.client = client

    def search(self, prompt: str, req=None) -> Dict[str, Any]:
        if self.client is None:
            return self._demo_narrative(req)
        try:
            response = self.client.messages.create(
                model=Config.get_model(),
                max_tokens=Config.get_max_tokens(),
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=[{"role": "user", "content": prompt}],
            )
            answer_parts: List[str] = []
            sources: List[Dict[str, str]] = []
            seen_urls = set()

            for block in response.content:
                btype = getattr(block, "type", None)
                if btype == "text":
                    answer_parts.append(block.text)
                    for cite in (getattr(block, "citations", None) or []):
                        url = getattr(cite, "url", None)
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            sources.append({"title": getattr(cite, "title", "") or url, "url": url})
                elif btype == "web_search_tool_result":
                    content = getattr(block, "content", None) or []
                    for item in content:
                        url = getattr(item, "url", None)
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            sources.append({"title": getattr(item, "title", "") or url, "url": url})

            answer = "\n".join(answer_parts).strip() or "No data available"
            return {"answer": answer, "sources": sources[:5], "timestamp": datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"Market intel failed: {e}")
            return {"answer": f"Search unavailable: {e}", "sources": [], "error": str(e)}

    def _demo_narrative(self, req) -> Dict[str, Any]:
        """Canned, domain-appropriate narrative for offline/demo mode - no live
        web search available without an API key."""
        if req is None:
            return {"answer": "Demo mode: no live market search available.",
                    "sources": [], "timestamp": datetime.now().isoformat()}
        from domains import DOMAINS  # local import avoids a circular import at module load

        vendor_db = DOMAINS[req.domain]["vendors"]
        matching = [v for v, d in vendor_db.items() if req.workload in d.get("workloads", [])]
        leaders = matching[:4] or list(vendor_db.keys())[:4]
        year = datetime.now().year
        answer = (
            f"[Demo mode - offline fixture, no live web search] The {year} enterprise "
            f"{req.domain.lower()} market for {req.workload} workloads remains led by "
            f"{', '.join(leaders[:-1])} and {leaders[-1]}" if len(leaders) > 1 else
            f"[Demo mode - offline fixture, no live web search] The {year} enterprise "
            f"{req.domain.lower()} market for {req.workload} workloads is led by {leaders[0]}"
        )
        answer += (
            f". Deployment preferences continue to shift toward {req.deployment.lower()} models, "
            f"with vendors differentiating on price/performance, data reduction, and management "
            f"simplicity. Set ANTHROPIC_API_KEY to replace this fixture with a live, cited web search."
        )
        return {
            "answer": answer,
            "sources": [{"title": "Demo mode - offline fixture (no live web search)", "url": ""}],
            "timestamp": datetime.now().isoformat(),
        }


# ==========================================================
# AI Analysis — Claude JSON mode
# ==========================================================

class AIAnalyzer:
    def __init__(self, client: Optional[anthropic.Anthropic]):
        self.client = client

    def _generate_json(self, system: str, prompt: str) -> Optional[dict]:
        """JSON call with one retry. Claude has no forced-JSON response mode like
        Gemini's response_mime_type, so we instruct strictly in the system prompt
        and strip any code fences defensively before parsing."""
        json_system = (
            system + "\n\nRespond with ONLY valid JSON. No markdown code fences, "
            "no preamble, no explanation outside the JSON object."
        )
        raw = ""
        for attempt in (1, 2):
            try:
                response = self.client.messages.create(
                    model=Config.get_model(),
                    max_tokens=Config.get_max_tokens(),
                    system=json_system,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = "".join(
                    b.text for b in response.content if getattr(b, "type", None) == "text"
                )
                if not raw:
                    logger.error(f"Empty Claude response (attempt {attempt}), "
                                 f"stop_reason={getattr(response, 'stop_reason', None)}")
                    continue
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL)
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse failed (attempt {attempt}): {e} | raw: {raw[:300]}")
                continue
            except Exception as e:
                logger.error(f"Claude call failed (attempt {attempt}): {e}")
                continue
        return None

    def analyze_architecture(self, req, market_data: Dict) -> Dict:
        if self.client is None:
            return self._demo_architecture(req)
        metrics = ", ".join(f"{k}: {v}" for k, v in req.metrics().items()) or "N/A"
        market_insight = (market_data.get("answer") or "No market data")[:800]

        prompt = f"""As an enterprise {req.domain.lower()} architect, analyze these requirements:

Domain: {req.domain}
Workload: {req.workload}
Scale: {req.capacity} {req.unit}
Deployment: {req.deployment}
Priority: {req.priority}
Availability SLA: {req.availability_target}
Performance Profile: {metrics}

Market Context: {market_insight}

Return JSON with exactly these keys:
- architecture_type (string - the recommended architecture pattern for this domain)
- key_recommendations (array of 3-4 concise strings)
- performance_requirements (object)
- scalability_notes (string)
- redundancy_approach (string)"""

        result = self._generate_json(
            system=f"You are an enterprise {req.domain.lower()} architect. Return only valid JSON.",
            prompt=prompt)
        if not result or "architecture_type" not in result:
            return {"architecture_type": "Unknown",
                    "key_recommendations": ["Analysis failed - please retry"],
                    "error": "Invalid or empty AI response"}
        result.setdefault("key_recommendations", [])
        return result

    def evaluate_vendors(self, req, vendors: List[str], market_data: Dict,
                          procurement_context: Optional[List[Dict]] = None) -> List[Dict]:
        from domains import DOMAINS  # local import avoids a circular import at module load

        if not vendors:
            return []
        vendor_db = DOMAINS[req.domain]["vendors"]
        if self.client is None:
            return self._demo_vendor_ranking(req, vendors, vendor_db, procurement_context)
        vendor_info = []
        for v in vendors[:8]:
            data = vendor_db.get(v)
            if not data:
                continue
            vendor_info.append(
                f"- {v}: Strengths: {', '.join(data['strengths'])} | Cost: {data.get('cost_profile', 'N/A')}"
                + (f" | Best for: {data['sweet_spot']}" if data.get("sweet_spot") else "")
            )
        if not vendor_info:
            return []

        proc_section = ""
        if procurement_context:
            chunks = [f"[{d['id']}] {d['text'][:400]}" for d in procurement_context[:3]]
            proc_section = ("\nInternal Procurement Context (RAG - retrieved from company documents):\n"
                             + "\n\n".join(chunks) + "\n")

        prompt = f"""Evaluate these {req.domain.lower()} vendors for the following requirements:

Domain: {req.domain}
Workload: {req.workload}
Scale: {req.capacity} {req.unit}
Deployment: {req.deployment}
Priority: {req.priority}
Availability SLA: {req.availability_target}

Available Vendors:
{chr(10).join(vendor_info)}
{proc_section}
Return JSON with an array 'vendors' containing the top 3-4 vendors ranked by fit:
- name (string - must exactly match a name from Available Vendors)
- fit_score (number 1-10; consider workload match, scale fit, priority alignment, AND existing procurement agreements/policy from internal context)
- strengths (array of 2-3 strings specific to this use case; mention existing agreements where relevant)
- considerations (array of 2-3 strings; flag policy hurdles like new-vendor security review where relevant)

Prioritize vendors whose cost profile matches the stated priority, and weigh internal
procurement context into the ranking. Do NOT estimate costs - TCO is computed separately."""

        result = self._generate_json(
            system=f"You are an enterprise {req.domain.lower()} analyst. Return only JSON with a 'vendors' array.",
            prompt=prompt)
        if not result or not isinstance(result.get("vendors"), list):
            return [{"name": "Error", "fit_score": 0, "strengths": [],
                     "considerations": ["Vendor evaluation failed - invalid AI response"],
                     "error": "Invalid or empty AI response"}]
        return [v for v in result["vendors"] if isinstance(v, dict) and v.get("name")]

    def _demo_architecture(self, req) -> Dict:
        """Realistic architecture fixture for offline/demo mode, built from the
        domain's own workload metrics rather than a live Claude call."""
        metrics = req.metrics()
        ha_note = ("active-active clustering across sites" if "99.99" in req.availability_target
                    or "99.999" in req.availability_target else "active-passive failover within a site")
        return {
            "architecture_type": f"{req.workload} - optimized {req.domain.lower()} architecture "
                                  f"[Demo mode - offline fixture]",
            "key_recommendations": [
                f"Size for {req.capacity} {req.unit} with headroom for 30% organic growth",
                f"Deploy as {req.deployment.lower()} to match stated deployment preference",
                f"Prioritize {req.priority.lower()} in vendor and configuration selection",
                "Set ANTHROPIC_API_KEY to replace this fixture with live Claude reasoning",
            ],
            "performance_requirements": metrics or {"Note": "No published benchmark for this workload"},
            "scalability_notes": f"Scale-out approach recommended to reach {req.capacity} {req.unit} "
                                  f"while preserving the workload's latency/throughput profile.",
            "redundancy_approach": f"{ha_note.capitalize()} to meet the {req.availability_target} SLA target.",
        }

    def _demo_vendor_ranking(self, req, vendors: List[str], vendor_db: Dict,
                              procurement_context: Optional[List[Dict]]) -> List[Dict]:
        """Deterministic, rule-based vendor ranking for offline/demo mode - uses
        the same vendor metadata (strengths, cost profile, sweet spot) the live
        prompt would, just scored without an LLM call."""
        priority = (req.priority or "").lower()
        priority_cost_fit = {
            "cost": {"budget": 3, "competitive": 1, "premium": -2, "variable": 0},
            "performance": {"premium": 3, "competitive": 1, "budget": -1, "variable": 0},
        }
        cost_fit = next((v for k, v in priority_cost_fit.items() if k in priority), {})

        agreement_vendors = set()
        if procurement_context:
            for doc in procurement_context:
                text = (doc.get("text") or "").lower()
                for v in vendors:
                    if v.lower() in text:
                        agreement_vendors.add(v)

        scored = []
        for v in vendors[:8]:
            data = vendor_db.get(v)
            if not data:
                continue
            score = 6.0 + cost_fit.get(data.get("cost_profile", ""), 0)
            has_agreement = v in agreement_vendors
            if has_agreement:
                score += 1.5
            score = max(1.0, min(10.0, score))

            strengths = list(data.get("strengths", []))[:3]
            if has_agreement:
                strengths = strengths[:2] + ["Existing negotiated procurement agreement"]

            considerations = []
            if data.get("sweet_spot"):
                considerations.append(f"Sweet spot: {data['sweet_spot']}")
            if not has_agreement:
                considerations.append("No existing procurement agreement - may need new-vendor security review")

            scored.append({
                "name": v, "fit_score": round(score, 1),
                "strengths": strengths or ["Meets workload requirements"],
                "considerations": considerations or ["No specific concerns identified"],
            })

        scored.sort(key=lambda x: x["fit_score"], reverse=True)
        return scored[:4]
