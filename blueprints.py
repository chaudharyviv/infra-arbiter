"""
Solution Blueprints - cross-domain correlation layer.

A blueprint maps a business use case to multiple infrastructure domains with
DETERMINISTIC sizing formulas linked to one driver metric. Example: an AI/ML
training platform sized by GPU server count derives its dataset storage and
metadata database capacity from that same driver - the domains are correlated,
not analyzed in isolation.

The existing LangGraph workflow runs once per component (unchanged); this layer
derives the inputs, then aggregates outputs: combined TCO, cross-domain vendor
synergy, and stack-level compliance posture.
"""

from typing import Dict, Any, List, Callable
from dataclasses import dataclass


@dataclass
class Component:
    domain: str
    workload: str
    size_fn: Callable[[int, Dict[str, float]], int]   # (driver, params) -> capacity
    rationale: str                                     # template; {param} placeholders allowed


# Each blueprint exposes its sizing ASSUMPTIONS as named, user-editable
# parameters. The formulas stay deterministic; the inputs are transparent
# and adjustable in the UI - nothing hidden in code.
BLUEPRINTS: Dict[str, Dict[str, Any]] = {

    "AI/ML Training Platform": {
        "description": "GPU compute + high-throughput dataset storage + experiment/metadata store, "
                       "sized from the number of GPU servers.",
        "driver": {"label": "GPU Servers", "min": 2, "max": 64, "default": 8, "step": 2},
        "params": {
            "tb_per_gpu_server": {"label": "Storage TB per GPU server", "default": 40, "min": 10, "max": 120, "step": 5,
                                  "help": "Datasets + checkpoints + staging per GPU node to avoid I/O stall"},
            "metadata_tb_per_2_servers": {"label": "Metadata TB per 2 GPU servers", "default": 1, "min": 1, "max": 10, "step": 1,
                                          "help": "Experiment tracking / feature metadata store sizing"},
        },
        "components": [
            Component("Server / Compute", "AI/GPU Compute",
                      lambda n, p: n,
                      "Driver metric - one entry per GPU server (e.g., 8×GPU HGX class)."),
            Component("Storage", "AI/ML Training",
                      lambda n, p: int(n * p["tb_per_gpu_server"]),
                      "{tb_per_gpu_server}TB high-performance storage per GPU server "
                      "(datasets + checkpoints + staging) to keep GPUs fed."),
            Component("Database", "NoSQL / Document",
                      lambda n, p: max(2, int((n // 2) * p["metadata_tb_per_2_servers"])),
                      "Experiment/metadata store: {metadata_tb_per_2_servers}TB per 2 GPU servers, 2TB floor."),
        ],
    },

    "Core Banking Modernization": {
        "description": "OLTP database + dedicated DB hosts + resilient block storage + "
                       "transactional messaging, sized from primary database size.",
        "driver": {"label": "Primary DB Size (TB)", "min": 2, "max": 100, "default": 10, "step": 2},
        "params": {
            "storage_multiple": {"label": "Storage multiple of DB size", "default": 4, "min": 2, "max": 8, "step": 1,
                                 "help": "Sync replica + backup staging + growth headroom (Tier-1 standard: 4×)"},
            "hosts_per_5tb": {"label": "DB hosts per 5TB (HA pairs)", "default": 2, "min": 1, "max": 4, "step": 1,
                              "help": "High-memory hosts per 5TB; 4-node floor across sites"},
            "msg_instances_per_2tb": {"label": "Messaging instances per 2TB", "default": 1, "min": 1, "max": 4, "step": 1,
                                      "help": "Transactional messaging scaled with throughput proxy; 4-instance floor"},
        },
        "components": [
            Component("Database", "OLTP / Core Banking",
                      lambda d, p: d,
                      "Driver metric - primary transactional data set."),
            Component("Server / Compute", "Database Hosts",
                      lambda d, p: max(4, int((d // 5) * p["hosts_per_5tb"])),
                      "{hosts_per_5tb} high-memory hosts per 5TB for HA pairs, 4-node floor "
                      "(primary + standby across sites)."),
            Component("Storage", "OLTP Database",
                      lambda d, p: int(d * p["storage_multiple"]),
                      "{storage_multiple}× primary size: synchronous replica + backup staging + "
                      "growth headroom, per Tier-1 resilience standards."),
            Component("Middleware", "Messaging / Streaming",
                      lambda d, p: max(4, int((d // 2) * p["msg_instances_per_2tb"])),
                      "{msg_instances_per_2tb} messaging instance(s) per 2TB throughput proxy, "
                      "4-instance floor for quorum + HA."),
        ],
    },

    "Enterprise Data Lake & Analytics": {
        "description": "Object/file storage for the lake + analytics database + ingestion streaming, "
                       "sized from raw data volume.",
        "driver": {"label": "Raw Data Volume (TB)", "min": 50, "max": 2000, "default": 400, "step": 50},
        "params": {
            "lake_multiple_pct": {"label": "Lake storage % of raw volume", "default": 150, "min": 100, "max": 300, "step": 10,
                                  "help": "Landing + curated zones; compression offsets copies (default 150%)"},
            "warehouse_pct": {"label": "Warehouse % of raw volume", "default": 10, "min": 5, "max": 30, "step": 5,
                              "help": "Share of raw volume materialized into the analytics layer"},
            "stream_per_100tb": {"label": "Streaming instances per 100TB", "default": 1, "min": 1, "max": 5, "step": 1,
                                 "help": "Ingestion cluster sizing; 4-instance floor"},
        },
        "components": [
            Component("Storage", "File Services",
                      lambda d, p: int(d * p["lake_multiple_pct"] / 100),
                      "{lake_multiple_pct}% of raw volume: landing + curated zones."),
            Component("Database", "OLAP / Analytics",
                      lambda d, p: max(5, int(d * p["warehouse_pct"] / 100)),
                      "{warehouse_pct}% of raw volume materialized into the analytics/warehouse layer."),
            Component("Middleware", "Messaging / Streaming",
                      lambda d, p: max(4, int((d // 100) * p["stream_per_100tb"])),
                      "{stream_per_100tb} ingestion instance(s) per 100TB raw volume, 4-instance floor."),
        ],
    },

    # ------------------------------------------------------------------
    # Payments / Real-Time Transaction Platform
    # Messaging-primary, strict Tier-1, latency-sensitive — distinct from
    # Core Banking (which is DB-primary). Shows Middleware depth + Tier-1
    # compliance pressure across the stack.
    # ------------------------------------------------------------------
    "Payments / Real-Time Transaction Platform": {
        "description": "Low-latency messaging fabric + payments ledger database + resilient "
                       "block storage, sized from peak transactions-per-second. Messaging is "
                       "the primary domain; Tier-1 SLA (99.999%) is assumed for the whole stack.",
        "driver": {"label": "Peak TPS (thousands)", "min": 5, "max": 200, "default": 50, "step": 5},
        "params": {
            "msg_instances_per_10k_tps": {
                "label": "Messaging instances per 10K TPS", "default": 2, "min": 1, "max": 6, "step": 1,
                "help": "Broker / stream partition capacity; 6-instance floor for multi-AZ quorum",
            },
            "ledger_tb_per_20k_tps": {
                "label": "Ledger DB TB per 20K TPS", "default": 2, "min": 1, "max": 10, "step": 1,
                "help": "Hot ledger + settlement history retention proxy",
            },
            "storage_multiple": {
                "label": "Storage multiple of ledger size", "default": 5, "min": 3, "max": 10, "step": 1,
                "help": "Sync replica + async DR copy + audit staging (Tier-1 payments standard: 5×)",
            },
        },
        "components": [
            Component("Middleware", "Messaging / Streaming",
                      lambda tps, p: max(6, int((tps // 10) * p["msg_instances_per_10k_tps"])),
                      "{msg_instances_per_10k_tps} messaging instance(s) per 10K TPS, "
                      "6-instance floor for multi-AZ quorum + exactly-once producers."),
            Component("Database", "OLTP / Core Banking",
                      lambda tps, p: max(2, int((tps // 20) * p["ledger_tb_per_20k_tps"])),
                      "Payments ledger: {ledger_tb_per_20k_tps}TB per 20K TPS peak, 2TB floor."),
            Component("Storage", "OLTP Database",
                      lambda tps, p: max(
                          10,
                          int(max(2, int((tps // 20) * p["ledger_tb_per_20k_tps"])) * p["storage_multiple"]),
                      ),
                      "{storage_multiple}× ledger size: synchronous replica + async DR + "
                      "audit staging, per Tier-1 payments resilience standards."),
        ],
    },

    # ------------------------------------------------------------------
    # Hybrid Cloud Landing Zone / VDI Platform
    # Server + Storage (+ optional thin Middleware for profile services).
    # Emphasises hybrid deployment, VDI boot-storm storage, and vendor
    # concentration across compute and storage.
    # ------------------------------------------------------------------
    "Hybrid Cloud Landing Zone / VDI Platform": {
        "description": "Virtualization / VDI hosts + user-profile and golden-image storage, "
                       "sized from concurrent desktop count. Hybrid-friendly; surfaces "
                       "cross-domain concentration when the same vendor wins compute and storage.",
        "driver": {"label": "Concurrent Desktops", "min": 100, "max": 10000, "default": 2000, "step": 100},
        "params": {
            "users_per_host": {
                "label": "Users per virtualization host", "default": 50, "min": 20, "max": 120, "step": 5,
                "help": "Density depends on desktop image weight and GPU offload; 50 is a typical non-GPU baseline",
            },
            "profile_gb_per_user": {
                "label": "Profile + golden-image GB per user", "default": 25, "min": 10, "max": 80, "step": 5,
                "help": "Roaming profile + AppVolumes / FSLogix style layers + golden image share",
            },
            "boot_storm_factor": {
                "label": "Boot-storm capacity factor", "default": 1.5, "min": 1.0, "max": 2.5, "step": 0.1,
                "help": "Extra storage headroom so morning login storms stay within latency SLO",
            },
        },
        "components": [
            Component("Server / Compute", "Virtualization Hosts",
                      lambda n, p: max(4, int(n / p["users_per_host"])),
                      "Virtualization hosts: {users_per_host} concurrent users per host, "
                      "4-node floor for HA across racks / sites."),
            Component("Storage", "VDI",
                      lambda n, p: max(
                          20,
                          int((n * p["profile_gb_per_user"] / 1000) * p["boot_storm_factor"]),
                      ),
                      "VDI storage: {profile_gb_per_user}GB profile/image per user × "
                      "{boot_storm_factor} boot-storm factor, 20TB floor."),
        ],
    },

    # ------------------------------------------------------------------
    # Disaster Recovery / Secondary Site
    # Capacity is a fraction of a named primary footprint — not a new
    # workload type. Demonstrates asymmetric cost (DR is cheaper but
    # still multi-domain) and forces the reviewer to think about RPO/RTO
    # driven sizing rather than production peak.
    # ------------------------------------------------------------------
    "Disaster Recovery / Secondary Site": {
        "description": "Secondary-site compute, storage and database capacity derived as a "
                       "recoverable fraction of the primary footprint. Sized from primary "
                       "production capacity (TB equivalent); cost is intentionally asymmetric "
                       "to production (warm/pilot-light patterns).",
        "driver": {"label": "Primary Footprint (TB equiv.)", "min": 20, "max": 2000, "default": 200, "step": 20},
        "params": {
            "dr_storage_pct": {
                "label": "DR storage % of primary", "default": 100, "min": 40, "max": 120, "step": 10,
                "help": "100% = full replica; 40–60% common for tiered / compressed DR copies",
            },
            "dr_compute_pct": {
                "label": "DR compute % of primary hosts", "default": 40, "min": 20, "max": 100, "step": 10,
                "help": "Pilot-light / warm standby ratio; 40% covers critical tier with scaled-down non-critical",
            },
            "dr_db_pct": {
                "label": "DR database % of primary", "default": 100, "min": 50, "max": 100, "step": 10,
                "help": "Usually full replica for RPO≈0 tiers; lower only when RPO allows log shipping lag",
            },
            "primary_hosts_per_50tb": {
                "label": "Primary hosts per 50TB (for DR calc)", "default": 4, "min": 2, "max": 12, "step": 1,
                "help": "Used only to derive DR host count from the TB driver",
            },
        },
        "components": [
            Component("Storage", "Backup",
                      lambda d, p: max(10, int(d * p["dr_storage_pct"] / 100)),
                      "DR storage: {dr_storage_pct}% of primary footprint "
                      "(full replica or compressed/tiered copy)."),
            Component("Server / Compute", "Virtualization Hosts",
                      lambda d, p: max(
                          2,
                          int((d / 50) * p["primary_hosts_per_50tb"] * p["dr_compute_pct"] / 100),
                      ),
                      "DR compute: {dr_compute_pct}% of primary host count "
                      "({primary_hosts_per_50tb} hosts per 50TB primary), 2-node floor."),
            Component("Database", "OLTP / Core Banking",
                      lambda d, p: max(1, int(d * p["dr_db_pct"] / 100 * 0.15)),
                      # 0.15 ≈ typical DB share of a mixed primary footprint when driver is TB-equiv
                      "DR database: {dr_db_pct}% of estimated primary DB share "
                      "(~15% of TB-equivalent footprint), 1TB floor."),
        ],
    },
}


def default_params(blueprint_name: str) -> Dict[str, float]:
    return {k: v["default"] for k, v in BLUEPRINTS[blueprint_name].get("params", {}).items()}


def derive_components(blueprint_name: str, driver_value: int,
                      params: Dict[str, float] = None) -> List[Dict[str, Any]]:
    """Resolve a blueprint into concrete per-domain requirements.
    params: user-adjusted sizing assumptions (defaults if None)."""
    bp = BLUEPRINTS[blueprint_name]
    p = {**default_params(blueprint_name), **(params or {})}
    out = []
    for c in bp["components"]:
        out.append({
            "domain": c.domain,
            "workload": c.workload,
            "capacity": c.size_fn(driver_value, p),
            "rationale": c.rationale.format(**p),
        })
    return out


def analyze_synergy(component_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cross-domain vendor correlation over per-component results.

    component_results: [{domain, workload, capacity, unit, state(final AgentState dict)}]

    Finds vendors recommended in more than one component (single-vendor
    leverage / bundle opportunity) and flags stack-level concentration.
    Deterministic - no LLM.
    """
    vendor_hits: Dict[str, List[str]] = {}
    vendor_spend: Dict[str, int] = {}
    total_spend = 0

    for comp in component_results:
        state = comp["state"]
        tco_by_vendor = {t["vendor"]: t for t in state.get("tco_estimates", [])}
        for rec in state.get("vendor_recommendations", [])[:3]:
            name = rec.get("name", "")
            if not name or rec.get("fit_score", 0) == 0:
                continue
            # Only count vendors validated against this domain's DB - a TCO entry
            # exists iff the name matched the registry (guards against LLM drift)
            tco = tco_by_vendor.get(name)
            if not tco:
                continue
            # Normalize umbrella vendors across domains (Dell == Dell, GCP* == GCP family)
            family = name.split(" (")[0].split(" / ")[0]
            for prefix in ("AWS", "Azure", "GCP", "Google"):
                if family.startswith(prefix):
                    family = "GCP" if prefix == "Google" else prefix
            vendor_hits.setdefault(family, []).append(f"{comp['domain']}: {name}")
            vendor_spend[family] = vendor_spend.get(family, 0) + tco["total_3yr"]
        # Total = top-1 vendor TCO per component (the presumptive selection)
        top = next((r for r in state.get("vendor_recommendations", []) if r.get("fit_score", 0) > 0), None)
        if top:
            t = tco_by_vendor.get(top.get("name", ""))
            if t:
                total_spend += t["total_3yr"]

    multi_domain = {v: hits for v, hits in vendor_hits.items() if len(set(h.split(":")[0] for h in hits)) > 1}

    concentration = []
    if total_spend:
        for v, spend in vendor_spend.items():
            share = spend / max(total_spend, 1)
            if v in multi_domain and share > 0.5:
                concentration.append(
                    f"{v} spans {len(set(h.split(':')[0] for h in multi_domain[v]))} domains at a large "
                    f"spend share - single-vendor leverage available, but review concentration limits")

    return {
        "multi_domain_vendors": multi_domain,
        "bundle_opportunities": [
            f"{v} appears across {', '.join(sorted(set(h.split(':')[0] for h in hits)))} - "
            f"negotiate a bundled/stack agreement instead of per-domain deals"
            for v, hits in multi_domain.items()
        ],
        "concentration_notes": concentration,
        "estimated_stack_tco_3yr": total_spend,
    }
