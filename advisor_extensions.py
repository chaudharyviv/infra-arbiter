"""
Procurement RAG + Deterministic TCO Engine
==========================================

ProcurementRAG
  In-memory retrieval over the local procurement corpus using TF-IDF +
  cosine similarity. Zero external embedding service, fully deterministic,
  zero extra latency/cost. Designed so the only public methods the graph
  nodes ever call are `retrieve()` and `negotiated_discounts()` - swap the
  backend for Voyage/pgvector/Chroma later without touching callers.

  Also extracts:
  - negotiated discounts (domain-scoped, expiry-aware)
  - realized historical unit costs from PO history documents
  - agreement freshness metadata for compliance / UI badges

TCOEngine
  Pure-Python 3-year total cost of ownership model. Pricing is a business
  decision; it is never delegated to an LLM. Rates are indicative list-price
  approximations; negotiated discounts are applied from the procurement
  corpus frontmatter. Domain-specific overrides live in domains.py.

  Includes a sensitivity() helper that produces Base / Optimistic /
  Conservative scenarios from any base estimate - still pure functions.

Design intent for portfolio / enterprise demos:
  - Completely offline-capable
  - Auditable formulas
  - Transparent uncertainty bands
  - Easy to extend with more sophisticated cost drivers
"""

from __future__ import annotations

import os
import re
import glob
import logging
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("infra-advisor.ext")


def _safe_max_df(n_docs: int, min_df: int = 1, desired_max_df: float = 0.95) -> float:
    """TfidfVectorizer raises when floor(max_df * n_docs) < min_df - which
    happens whenever a domain-filtered pool has only a handful of documents
    (e.g. a single-vendor domain). Fall back to no upper bound in that case."""
    if n_docs <= 1 or int(desired_max_df * n_docs) < min_df:
        return 1.0
    return desired_max_df

DATA_DIR = os.environ.get(
    "PROCUREMENT_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "procurement"),
)


# ---------------------------------------------------------------------------
# Frontmatter + corpus loading
# ---------------------------------------------------------------------------

def _parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """Parse a simple YAML-like '---' frontmatter block.
    Returns (metadata_dict, body_text).
    """
    meta: Dict[str, str] = {}
    body = text
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        body = m.group(2)
    return meta, body


def load_corpus(data_dir: str = DATA_DIR) -> List[Dict[str, Any]]:
    """Load every .md document under data_dir.

    Each document becomes one retrieval chunk (docs are intentionally short
    and self-contained). Frontmatter supplies structured metadata used by
    both retrieval filtering and the discount engine.
    """
    docs: List[Dict[str, Any]] = []
    if not os.path.isdir(data_dir):
        logger.warning("Procurement data directory not found: %s", data_dir)
        return docs

    for path in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta, body = _parse_frontmatter(f.read())
            docs.append({
                "id": os.path.basename(path),
                "meta": meta,
                "text": body.strip(),
                "path": path,
            })
        except Exception as exc:
            logger.error("Failed to load %s: %s", path, exc)

    logger.info("Loaded %d procurement documents from %s", len(docs), data_dir)
    return docs


def _parse_valid_until(raw: str) -> Optional[date]:
    """Parse YYYY-MM-DD (or YYYY-MM) into a date. Returns None on failure."""
    if not raw:
        return None
    raw = raw.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Realized-cost extraction from PO history documents
# ---------------------------------------------------------------------------

# Patterns that appear in the shipped po_history_2024_2025.md and similar docs.
_PO_LINE_RE = re.compile(
    r"PO-\d+-\d+:\s*(?P<vendor>[\w\s/\(\)]+?)\s+.*?,"
    r"\s*(?P<capacity>[\d,]+)\s*TB.*?,"
    r"\s*\$(?P<cost>[\d,]+)",
    re.IGNORECASE,
)
_AVG_ONPREM_RE = re.compile(
    r"Average realized cost across on-prem purchases:\s*~?\$?([\d,]+)/TB",
    re.IGNORECASE,
)
_AVG_OBJECT_RE = re.compile(
    r"Object storage realized cost:\s*~?\$?([\d,]+)/TB",
    re.IGNORECASE,
)


def extract_realized_costs(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Parse purchase-order history documents into structured realized costs.

    Returns a dict shaped for easy UI / ARB consumption:
    {
      "by_vendor": {"NetApp": {"total_spend": ..., "total_tb": ..., "per_tb": ...}, ...},
      "onprem_avg_per_tb": 1450,
      "object_avg_per_tb": 370,
      "po_count": 5,
      "source_docs": ["po_history_2024_2025.md"],
    }
    """
    by_vendor: Dict[str, Dict[str, float]] = {}
    onprem_avg: Optional[float] = None
    object_avg: Optional[float] = None
    po_count = 0
    source_docs: List[str] = []

    for d in docs:
        meta = d.get("meta", {})
        if meta.get("doc_type") not in ("purchase_order_history", "po_history"):
            # Also accept any doc whose body clearly contains PO lines
            if "PO-" not in d.get("text", ""):
                continue
        source_docs.append(d["id"])
        text = d.get("text", "")

        for m in _PO_LINE_RE.finditer(text):
            vendor_raw = m.group("vendor").strip()
            # Normalise common prefixes: "NetApp AFF ..." → "NetApp"
            vendor = vendor_raw.split()[0] if vendor_raw else "Unknown"
            # Map GCP / Google variants
            if vendor.lower().startswith("gcp") or vendor.lower().startswith("google"):
                vendor = "GCP"
            try:
                capacity = float(m.group("capacity").replace(",", ""))
                cost = float(m.group("cost").replace(",", ""))
            except ValueError:
                continue
            po_count += 1
            entry = by_vendor.setdefault(vendor, {"total_spend": 0.0, "total_tb": 0.0})
            entry["total_spend"] += cost
            entry["total_tb"] += capacity

        avg_m = _AVG_ONPREM_RE.search(text)
        if avg_m:
            try:
                onprem_avg = float(avg_m.group(1).replace(",", ""))
            except ValueError:
                pass
        obj_m = _AVG_OBJECT_RE.search(text)
        if obj_m:
            try:
                object_avg = float(obj_m.group(1).replace(",", ""))
            except ValueError:
                pass

    for v, entry in by_vendor.items():
        tb = entry["total_tb"]
        entry["per_tb"] = round(entry["total_spend"] / tb) if tb else 0

    return {
        "by_vendor": by_vendor,
        "onprem_avg_per_tb": onprem_avg,
        "object_avg_per_tb": object_avg,
        "po_count": po_count,
        "source_docs": source_docs,
    }


# ---------------------------------------------------------------------------
# ProcurementRAG
# ---------------------------------------------------------------------------

class ProcurementRAG:
    """Minimal, deterministic in-memory RAG over the procurement corpus.

    Retrieval is TF-IDF + cosine similarity, fit once at construction.
    Domain filtering prevents generic policy documents from drowning out
    vendor-specific agreements that share surface vocabulary.

    Extras beyond basic retrieve / discounts:
      - agreement expiry awareness (valid_until frontmatter)
      - realized historical unit costs from PO history
      - richer corpus_summary for diagnostics / UI
    """

    def __init__(
        self,
        client=None,
        docs: Optional[List[Dict[str, Any]]] = None,
        min_score: float = 0.05,
        reference_date: Optional[date] = None,
    ):
        self.client = client  # kept for interface stability
        self.docs = docs if docs is not None else load_corpus()
        self.min_score = min_score
        self.reference_date = reference_date or date.today()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self._realized_cache: Optional[Dict[str, Any]] = None

        if self.docs:
            # Include both body and key metadata tokens so vendor/domain
            # names participate in the vector space.
            corpus_texts = [
                f"{d['meta'].get('vendor', '')} {d['meta'].get('domain', '')} "
                f"{d['meta'].get('doc_type', '')} {d['text']}"
                for d in self.docs
            ]
            self.vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_df=_safe_max_df(len(corpus_texts)),
                min_df=1,
            )
            self.matrix = self.vectorizer.fit_transform(corpus_texts)
            logger.debug("TF-IDF matrix shape: %s", self.matrix.shape)

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        domain: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return the top-k most relevant documents with similarity scores.

        When `domain` is supplied, the candidate pool is restricted to
        documents tagged for that domain plus any untagged / cross-cutting
        documents (policy, multi-vendor history, etc.).
        """
        if not self.docs:
            return []

        pool = self.docs
        if domain:
            pool = [
                d for d in self.docs
                if not d["meta"].get("domain") or d["meta"]["domain"] == domain
            ]
            if not pool:
                pool = self.docs  # graceful fallback

        # Re-fit only when the pool differs from the full corpus
        if pool is self.docs and self.matrix is not None and self.vectorizer is not None:
            matrix, vectorizer = self.matrix, self.vectorizer
        else:
            texts = [
                f"{d['meta'].get('vendor', '')} {d['meta'].get('domain', '')} "
                f"{d['meta'].get('doc_type', '')} {d['text']}"
                for d in pool
            ]
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_df=_safe_max_df(len(texts)),
                min_df=1,
            )
            matrix = vectorizer.fit_transform(texts)

        q_vec = vectorizer.transform([query])
        scores = cosine_similarity(q_vec, matrix)[0]
        order = scores.argsort()[::-1][:top_k]

        results = []
        for i in order:
            if scores[i] <= self.min_score:
                continue
            hit = {**pool[i], "score": float(scores[i])}
            results.append(hit)

        return results

    # ------------------------------------------------------------------
    # Commercial terms
    # ------------------------------------------------------------------

    def agreement_status(
        self, domain: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Return rich agreement metadata per vendor.

        {
          "Dell": {
            "discount_pct": 18,
            "valid_until": date(2027, 12, 31),
            "days_remaining": 516,
            "status": "active" | "expiring_soon" | "expired",
            "doc_id": "dell_framework_agreement.md",
            "domain": "Storage",
          },
          ...
        }
        """
        out: Dict[str, Dict[str, Any]] = {}
        for d in self.docs:
            meta = d["meta"]
            vendor = meta.get("vendor", "").strip()
            pct = meta.get("discount_pct", "").strip()
            doc_domain = meta.get("domain", "").strip()

            if domain and doc_domain and doc_domain != domain:
                continue
            if not vendor or vendor.lower() in ("none", "multiple", ""):
                continue
            if not pct:
                continue
            try:
                discount_pct = float(pct)
            except ValueError:
                continue

            valid_until = _parse_valid_until(meta.get("valid_until", ""))
            days_remaining: Optional[int] = None
            status = "active"
            if valid_until:
                days_remaining = (valid_until - self.reference_date).days
                if days_remaining < 0:
                    status = "expired"
                elif days_remaining <= 180:
                    status = "expiring_soon"

            # Prefer the agreement with the highest discount if multiple
            existing = out.get(vendor)
            if existing and existing.get("discount_pct", 0) >= discount_pct:
                continue

            out[vendor] = {
                "discount_pct": discount_pct,
                "valid_until": valid_until,
                "days_remaining": days_remaining,
                "status": status,
                "doc_id": d["id"],
                "domain": doc_domain or domain,
            }
        return out

    def negotiated_discounts(
        self, domain: Optional[str] = None, include_expired: bool = False
    ) -> Dict[str, float]:
        """Parse discount_pct from frontmatter into a vendor → fraction map.

        Domain scoping prevents a Storage agreement from incorrectly
        applying to a Database evaluation (and vice-versa).

        By default expired agreements are excluded so TCO and preferred-vendor
        logic stay honest. Pass include_expired=True for audit views.
        """
        status = self.agreement_status(domain)
        discounts: Dict[str, float] = {}
        for vendor, info in status.items():
            if not include_expired and info["status"] == "expired":
                continue
            discounts[vendor] = info["discount_pct"] / 100.0
        return discounts

    def preferred_vendors(self, domain: Optional[str] = None) -> List[str]:
        """Vendors that currently hold an *active* (non-expired) agreement."""
        status = self.agreement_status(domain)
        return [
            v for v, info in status.items()
            if info["status"] in ("active", "expiring_soon")
        ]

    def realized_costs(self) -> Dict[str, Any]:
        """Lazy-parsed historical unit costs from PO history documents."""
        if self._realized_cache is None:
            self._realized_cache = extract_realized_costs(self.docs)
        return self._realized_cache

    def corpus_summary(self) -> Dict[str, Any]:
        """Lightweight introspection for UI / diagnostics."""
        by_domain: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        vendors = set()
        expiring: List[str] = []
        expired: List[str] = []

        status = self.agreement_status()
        for v, info in status.items():
            vendors.add(v)
            if info["status"] == "expiring_soon":
                expiring.append(v)
            elif info["status"] == "expired":
                expired.append(v)

        for d in self.docs:
            dom = d["meta"].get("domain") or "cross-cutting"
            by_domain[dom] = by_domain.get(dom, 0) + 1
            dtype = d["meta"].get("doc_type") or "unknown"
            by_type[dtype] = by_type.get(dtype, 0) + 1

        realized = self.realized_costs()
        return {
            "document_count": len(self.docs),
            "by_domain": by_domain,
            "by_type": by_type,
            "vendors_with_agreements": sorted(vendors),
            "expiring_soon": sorted(expiring),
            "expired": sorted(expired),
            "realized_po_count": realized.get("po_count", 0),
            "realized_onprem_avg_per_tb": realized.get("onprem_avg_per_tb"),
        }


# ---------------------------------------------------------------------------
# Deterministic TCO Engine
# ---------------------------------------------------------------------------

class TCOEngine:
    """3-year Total Cost of Ownership model.

    Design principles
    -----------------
    * Pure functions - no side effects, no LLM involvement.
    * Transparent formula: list rate × capacity × (1 − discount) + facilities + migration.
    * Uncertainty band (±15 %) reflects sizing / configuration variance.
    * Domain overrides arrive via `tco_cfg` so the same engine serves Storage,
      Server, Database and Middleware without hard-coded special cases.
    * sensitivity() produces Base / Optimistic / Conservative scenarios from
      any base estimate without re-running the full model.
    """

    # Default storage-oriented rates (overridden by domains.py for other domains)
    LIST_RATES = {
        "premium": 2600,
        "competitive": 1800,
        "budget": 1100,
    }
    CLOUD_MONTHLY_TIERS = [
        (100, 26.0),
        (500, 20.0),
        (float("inf"), 15.0),
    ]
    ONPREM_FACILITIES_PER_TB_3YR = 180
    MIGRATION_FLAT = 25_000

    @classmethod
    def _cloud_rate(cls, capacity: int, tiers=None) -> float:
        tiers = tiers or cls.CLOUD_MONTHLY_TIERS
        for limit, rate in tiers:
            if capacity < limit:
                return rate
        return tiers[-1][1]

    @classmethod
    def estimate(
        cls,
        vendor_name: str,
        vendor_meta: Dict[str, Any],
        capacity: int,
        discounts: Dict[str, float],
        tco_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a structured 3-year TCO estimate for one vendor.

        Parameters
        ----------
        vendor_name : exact key from the domain registry
        vendor_meta : the corresponding dict from DOMAINS[domain]["vendors"]
        capacity    : scale in domain units (TB, servers, instances, …)
        discounts   : vendor → fraction map from ProcurementRAG
        tco_cfg     : optional domain-specific overrides
        """
        cfg = tco_cfg or {}
        list_rates = cfg.get("list_rates", cls.LIST_RATES)
        cloud_tiers = cfg.get("cloud_monthly_tiers", cls.CLOUD_MONTHLY_TIERS)
        facilities_per_unit = cfg.get("facilities_per_unit", cls.ONPREM_FACILITIES_PER_TB_3YR)
        migration_flat = cfg.get("migration_flat", cls.MIGRATION_FLAT)

        deployments = vendor_meta.get("deployment", [])
        is_cloud_only = deployments == ["cloud"]
        discount = discounts.get(vendor_name, 0.0)

        if is_cloud_only:
            monthly_rate = cls._cloud_rate(capacity, cloud_tiers)
            base = monthly_rate * capacity * 36
            facilities = 0
            migration = migration_flat
            model = "OpEx (pay-as-you-go, 3-yr run rate)"
        else:
            profile = vendor_meta.get("cost_profile", "competitive")
            rate = list_rates.get(profile, list_rates.get("competitive", 1800))
            base = rate * capacity
            facilities = facilities_per_unit * capacity
            migration = migration_flat
            model = "CapEx + support (3-yr)"

        discounted_base = base * (1.0 - discount)
        total = discounted_base + facilities + migration
        low, high = total * 0.85, total * 1.15

        return {
            "vendor": vendor_name,
            "model": model,
            "capacity": capacity,
            "list_base": round(base),
            "negotiated_discount_pct": round(discount * 100),
            "facilities": round(facilities),
            "migration": round(migration),
            "total_3yr": round(total),
            "range": (round(low), round(high)),
            "per_unit_3yr": round(total / capacity) if capacity else 0,
            "has_agreement": discount > 0,
            "cost_profile": vendor_meta.get("cost_profile", "n/a"),
        }

    @classmethod
    def sensitivity(
        cls,
        base: Dict[str, Any],
        discount_delta_pct: float = 5.0,
        capacity_growth: float = 1.20,
        facilities_factor: float = 1.10,
    ) -> Dict[str, Dict[str, Any]]:
        """Produce Base / Optimistic / Conservative scenarios from a base estimate.

        All adjustments are pure arithmetic on the already-computed components.
        This keeps the sensitivity path fully deterministic and free of any
        re-invocation of vendor metadata or the LLM.

        Optimistic  : deeper discount, no capacity growth, lower facilities
        Base        : the original estimate
        Conservative: shallower (or zero extra) discount, capacity growth,
                      higher facilities burden
        """
        list_base = base.get("list_base", 0)
        facilities = base.get("facilities", 0)
        migration = base.get("migration", 0)
        disc_pct = base.get("negotiated_discount_pct", 0) / 100.0
        capacity = base.get("capacity", 1) or 1

        def _scenario(label: str, d: float, cap_mult: float, fac_mult: float) -> Dict[str, Any]:
            adj_base = list_base * (1.0 - d) * cap_mult
            adj_fac = facilities * fac_mult * cap_mult
            total = adj_base + adj_fac + migration
            return {
                "label": label,
                "total_3yr": round(total),
                "range": (round(total * 0.85), round(total * 1.15)),
                "per_unit_3yr": round(total / (capacity * cap_mult)) if capacity else 0,
                "assumptions": {
                    "discount_pct": round(d * 100, 1),
                    "capacity_multiplier": cap_mult,
                    "facilities_multiplier": fac_mult,
                },
            }

        opt_disc = min(disc_pct + discount_delta_pct / 100.0, 0.45)
        cons_disc = max(disc_pct - discount_delta_pct / 100.0, 0.0)

        return {
            "optimistic": _scenario("Optimistic", opt_disc, 1.0, 1.0 / facilities_factor),
            "base": _scenario("Base", disc_pct, 1.0, 1.0),
            "conservative": _scenario(
                "Conservative", cons_disc, capacity_growth, facilities_factor
            ),
        }

    @classmethod
    def compare_to_realized(
        cls,
        estimate: Dict[str, Any],
        realized: Dict[str, Any],
        unit_label: str = "TB",
    ) -> Optional[Dict[str, Any]]:
        """Overlay model per-unit cost against historical realized unit cost.

        Returns None when no useful realized signal exists for this vendor /
        domain. Otherwise returns a comparison dict suitable for UI badges
        and ARB footnotes.
        """
        model_per = estimate.get("per_unit_3yr") or 0
        if not model_per:
            return None

        vendor = estimate.get("vendor", "")
        by_vendor = realized.get("by_vendor") or {}
        # Prefer vendor-specific realized figure; fall back to on-prem average
        vendor_entry = by_vendor.get(vendor)
        realized_per = None
        source = None
        if vendor_entry and vendor_entry.get("per_tb"):
            realized_per = vendor_entry["per_tb"]
            source = f"PO history ({vendor})"
        elif realized.get("onprem_avg_per_tb"):
            realized_per = realized["onprem_avg_per_tb"]
            source = "on-prem portfolio average"

        if not realized_per:
            return None

        delta_pct = ((model_per - realized_per) / realized_per) * 100.0
        return {
            "model_per_unit": model_per,
            "realized_per_unit": realized_per,
            "delta_pct": round(delta_pct, 1),
            "unit": unit_label,
            "source": source,
            "interpretation": (
                "model above recent realized"
                if delta_pct > 10
                else "model below recent realized"
                if delta_pct < -10
                else "model aligned with recent realized"
            ),
        }

    @staticmethod
    def fmt(n: float) -> str:
        """Human-readable currency formatting for UI and ARB documents."""
        if n >= 1_000_000:
            return f"${n / 1_000_000:.2f}M"
        if n >= 1_000:
            return f"${n / 1_000:.0f}K"
        return f"${n:.0f}"

    @classmethod
    def sensitivity_note(cls) -> str:
        return (
            "Figures are pre-quotation comparative estimates. "
            "±15 % band reflects sizing and configuration uncertainty. "
            "Optimistic / Conservative scenarios adjust discount depth, "
            "capacity growth and facilities burden; they are deterministic "
            "overlays, not additional LLM estimates. "
            "Actual commercial offers will differ after detailed design and negotiation."
        )
