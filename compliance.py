"""
Compliance guardrail engine - deterministic policy-as-code checks applied to
every recommended vendor. No LLM involvement by design: policy is code.

Each check returns status: "pass" | "warn" | "fail" with a detail string.
Rules are intentionally simple and transparent - auditable by an ARB.

Policy packs
------------
Hard-coded constants (Tier-1 workloads, approved regions, assessment lead
time, concentration threshold) live in a PolicyPack dataclass. The default
pack mirrors a typical Indian bank / RBI outsourcing posture. Future packs
(e.g. EU, US) can be swapped without touching the check logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Literal, TypedDict

CheckStatus = Literal["pass", "warn", "fail"]


class CheckResult(TypedDict):
    name: str
    status: CheckStatus
    detail: str


class ComplianceResult(TypedDict):
    vendor: str
    overall: CheckStatus
    checks: List[CheckResult]
    policy_pack: str


@dataclass(frozen=True)
class PolicyPack:
    """Immutable policy configuration. Swap the pack, not the check code."""

    name: str = "Bank India / RBI (default)"
    tier1_workloads: Set[str] = field(default_factory=lambda: {
        "OLTP / Core Banking",
        "OLTP Database",
        "Messaging / Streaming",
    })
    production_sla_minimum: str = "99.99%"
    tier1_sla: str = "99.999%"
    approved_regions: List[str] = field(default_factory=lambda: ["asia-south1", "asia-south2"])
    approved_regions_cmek: bool = True
    new_vendor_assessment_weeks: str = "8–12"
    concentration_warn_share: float = 0.60
    # Soft signal: agreements expiring inside this many days surface a warn
    agreement_expiry_warn_days: int = 180

    @property
    def approved_regions_note(self) -> str:
        regions = " / ".join(self.approved_regions)
        return f"{regions} with CMEK" if self.approved_regions_cmek else regions


# Module-level default used by the graph node unless a caller injects another.
# RBI outsourcing rules require regulated data to stay in India - do not add
# other jurisdictions' regions here. Swap in one of the packs below instead.
DEFAULT_POLICY = PolicyPack()

# Example packs for other jurisdictions. Swap via run_compliance_checks(policy=...);
# check logic is unchanged - only the region list (and any other constants) differ.
APAC_POLICY = PolicyPack(
    name="Bank APAC (example)",
    approved_regions=["asia-southeast1", "asia-northeast1", "australia-southeast1"],
)

EU_POLICY = PolicyPack(
    name="Bank EU / GDPR (example)",
    approved_regions=["europe-west1", "europe-west3", "europe-west4"],
)

US_POLICY = PolicyPack(
    name="Bank US (example)",
    approved_regions=["us-east1", "us-central1", "us-west1"],
)

# Lookup for UI/config-driven pack selection - keyed by short jurisdiction
# label, not policy.name (which stays the descriptive string shown on
# results/reports).
POLICY_PACKS: Dict[str, PolicyPack] = {
    "India / RBI": DEFAULT_POLICY,
    "APAC": APAC_POLICY,
    "EU": EU_POLICY,
    "US": US_POLICY,
}


def _cloud_only(meta: Dict[str, Any]) -> bool:
    return meta.get("deployment", []) == ["cloud"]


def run_compliance_checks(
    domain: str,
    workload: str,
    deployment: str,
    availability_target: str,
    vendor_recommendations: List[Dict[str, Any]],
    vendor_db: Dict[str, Any],
    preferred_vendors: List[str],
    tco_estimates: List[Dict[str, Any]],
    policy: Optional[PolicyPack] = None,
    agreement_status: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[ComplianceResult]:
    """Returns one result block per recommended vendor.

    Parameters
    ----------
    preferred_vendors : active (non-expired) agreement holders
    agreement_status  : optional rich map from ProcurementRAG.agreement_status()
                        - when present, expiry warnings are emitted
    policy            : PolicyPack; defaults to Bank India / RBI
    """
    policy = policy or DEFAULT_POLICY
    results: List[ComplianceResult] = []
    tco_by_vendor = {t["vendor"]: t for t in tco_estimates}
    agreement_status = agreement_status or {}

    for rec in vendor_recommendations:
        name = rec.get("name", "")
        meta = vendor_db.get(name)
        if not meta or rec.get("fit_score", 0) == 0:
            continue

        checks: List[CheckResult] = []

        # --- 1. Availability SLA floor ---
        if workload in policy.tier1_workloads:
            if availability_target == policy.tier1_sla:
                checks.append({
                    "name": "Tier-1 SLA floor",
                    "status": "pass",
                    "detail": f"{policy.tier1_sla} meets Tier-1 requirement for this workload",
                })
            else:
                checks.append({
                    "name": "Tier-1 SLA floor",
                    "status": "fail",
                    "detail": (
                        f"Tier-1 workload requires {policy.tier1_sla} with synchronous "
                        f"replication; selected SLA is {availability_target}"
                    ),
                })
        else:
            if availability_target in (policy.production_sla_minimum, policy.tier1_sla):
                checks.append({
                    "name": "Production SLA floor",
                    "status": "pass",
                    "detail": (
                        f"{availability_target} meets the "
                        f"{policy.production_sla_minimum} production minimum"
                    ),
                })
            else:
                checks.append({
                    "name": "Production SLA floor",
                    "status": "warn",
                    "detail": (
                        f"{availability_target} is below the "
                        f"{policy.production_sla_minimum} production minimum - "
                        "acceptable only for non-production tiers"
                    ),
                })

        # --- 2. Data residency ---
        if _cloud_only(meta) or deployment == "Cloud":
            checks.append({
                "name": "Data residency",
                "status": "warn",
                "detail": (
                    f"Cloud deployment: regulated data must remain in-region "
                    f"({policy.approved_regions_note}). Confirm region pinning "
                    f"and CMEK before design approval"
                ),
            })
        elif deployment == "Hybrid":
            checks.append({
                "name": "Data residency",
                "status": "warn",
                "detail": (
                    f"Hybrid deployment: on-premises components meet residency "
                    f"directly; the cloud-resident portion must remain in-region "
                    f"({policy.approved_regions_note}). Confirm region pinning and "
                    f"CMEK for that portion before design approval - lower exposure "
                    f"than a full cloud deployment"
                ),
            })
        else:
            checks.append({
                "name": "Data residency",
                "status": "pass",
                "detail": "On-premises deployment - residency requirement inherently met",
            })

        # --- 3. Vendor onboarding + agreement freshness ---
        if name in preferred_vendors:
            info = agreement_status.get(name, {})
            status = info.get("status", "active")
            days = info.get("days_remaining")
            if status == "expiring_soon" and days is not None:
                checks.append({
                    "name": "Vendor onboarding",
                    "status": "warn",
                    "detail": (
                        f"Active agreement on file but expires in {days} days "
                        f"({info.get('valid_until')}). Plan renewal or dual-vendor "
                        f"coverage before procurement"
                    ),
                })
            else:
                checks.append({
                    "name": "Vendor onboarding",
                    "status": "pass",
                    "detail": (
                        "Active agreement on file - preferred vendor, no new "
                        "security assessment required"
                    ),
                })
        else:
            checks.append({
                "name": "Vendor onboarding",
                "status": "warn",
                "detail": (
                    f"No active agreement found. New vendors require security "
                    f"assessment ({policy.new_vendor_assessment_weeks} weeks) and "
                    f"RBI outsourcing compliance review before PO issuance"
                ),
            })

        # --- 4. Concentration risk ---
        tco = tco_by_vendor.get(name)
        if tco and tco_estimates:
            total = sum(t["total_3yr"] for t in tco_estimates) or 1
            share = tco["total_3yr"] / total
            if name in preferred_vendors and share > policy.concentration_warn_share:
                checks.append({
                    "name": "Concentration risk",
                    "status": "warn",
                    "detail": (
                        f"Vendor represents {share:.0%} of evaluated spend and "
                        f"already holds agreements - consider dual-vendor strategy "
                        f"per concentration guidelines (threshold "
                        f"{policy.concentration_warn_share:.0%})"
                    ),
                })
            else:
                checks.append({
                    "name": "Concentration risk",
                    "status": "pass",
                    "detail": "No concentration concern at this spend share",
                })

        overall: CheckStatus = (
            "fail" if any(c["status"] == "fail" for c in checks)
            else "warn" if any(c["status"] == "warn" for c in checks)
            else "pass"
        )
        results.append({
            "vendor": name,
            "overall": overall,
            "checks": checks,
            "policy_pack": policy.name,
        })

    return results
