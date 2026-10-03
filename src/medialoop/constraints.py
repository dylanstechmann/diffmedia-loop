"""Shared planner constraints exported by cell-protocol-compiler.

The compiler encodes published factor windows, required gates and mutually
exclusive alternatives. This module loads a constraints export (or a JSON
bundle of several), checks candidate tables against it, and reports drift
between the export and this package's built-in simulation box.

It never widens a window. When a candidate sits outside a published window,
or combines mutually exclusive alternatives, the violation is reported
instead of being clipped away.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from medialoop.space import FACTORS, NAMES

SUPPORTED_SCHEMA_VERSIONS = (1,)
AT_MOST_ONE_POSITIVE = "at_most_one_positive"


def check_candidates(rows, constraints: dict) -> list[str]:
    """Return human-readable violations of the exported constraints.

    Required parameter columns must be present in every row. Optional columns
    are checked when supplied. Exclusivity rules are evaluated on rows that
    carry all their members; a missing member column is itself a violation
    because load_bundle requires rule members to be required parameters.
    """
    violations: list[str] = []
    factors = constraints["factors"]
    for row in rows:
        candidate = row.get("candidate_id", "?")
        for name, window in factors.items():
            if name not in row or row[name] in (None, ""):
                if window.get("required"):
                    violations.append(f"{candidate}: missing required parameter {name}")
                continue
            try:
                value = float(row[name])
            except (TypeError, ValueError):
                violations.append(f"{candidate}: {name}={row[name]!r} is not numeric")
                continue
            inactive_value = window.get("inactive_value")
            is_inactive = inactive_value is not None and math.isclose(
                value, inactive_value, rel_tol=0.0, abs_tol=1e-9
            )
            if (not math.isfinite(value) or (not is_inactive and
                    not window["low"] - 1e-9 <= value <= window["high"] + 1e-9)):
                violations.append(
                    f"{candidate}: {name}={value:g} is outside the published window "
                    f"{window['low']:g}-{window['high']:g} {window.get('unit', '')} "
                    f"(from {window['protocol_id']})"
                )
        for rule in constraints["exclusivity"]:
            members = rule["parameters"]
            if not all(name in row and row[name] not in (None, "") for name in members):
                # The required-parameter pass already rejects a missing column
                # (load_bundle guarantees rule members are required), so an
                # omitted alternative cannot silently bypass the rule. The
                # rule itself is simply not evaluable on an incomplete row.
                continue
            try:
                values = [float(row[name]) for name in members]
            except (TypeError, ValueError):
                continue
            if sum(1 for value in values if value > 0) > 1:
                violations.append(
                    f"{candidate}: {', '.join(members)} are mutually exclusive "
                    "alternatives and cannot both be positive"
                )
            elif sum(1 for value in values if value > 0) < rule.get("minimum_positive", 0):
                violations.append(
                    f"{candidate}: one of {', '.join(members)} must be positive"
                )
    return violations


def factor_agreement(constraints: dict, *, require_complete: bool = False) -> list[str]:
    """Compare the bundle with the built-in simulation box.

    A published window that names a factor in the built-in box must sit
    inside that box: the box may be wider (it includes zero-dose corners for
    the cartoon), but it may not contradict a published window. A constrained
    factor the box does not carry (for example Noggin, whose BMP-pathway axis
    is simulated through LDN-193189 here) is not simulated but is still
    checked on candidate tables by ``check_candidates``; it is not drift.
    With ``require_complete`` every built-in box factor must also have an
    exported published window, so the simulation cannot quietly search a
    factor with no published bound. An empty list means agreement.
    """
    builtin = {item["name"]: item for item in FACTORS}
    discrepancies: list[str] = []
    for name, window in sorted(constraints["factors"].items()):
        box = builtin.get(name)
        if box is None:
            continue
        if window["low"] < box["low"] - 1e-9 or window["high"] > box["high"] + 1e-9:
            discrepancies.append(
                f"{name}: published window {window['low']:g}-{window['high']:g} is not "
                f"inside the simulation box {box['low']:g}-{box['high']:g}"
            )
    if require_complete:
        for name in NAMES:
            if name not in constraints["factors"]:
                discrepancies.append(
                    f"{name}: built-in simulation factor has no exported published window"
                )
    return discrepancies


def enforce_constraints(path, rows) -> dict:
    """Load a bundle, reject violating candidate rows, return provenance."""
    bundle = load_bundle(path)
    violations = check_candidates(rows, bundle)
    if violations:
        message = "; ".join(violations[:10])
        if len(violations) > 10:
            message += f" (and {len(violations) - 10} more)"
        raise ValueError(
            f"{path}: candidate table violates the exported published windows: {message}"
        )
    return {"sha256": bundle["source_sha256"], "protocols": bundle["protocols"]}


def _as_exports(payload) -> list:
    if isinstance(payload, list):
        return payload
    return [payload]


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_bundle(path) -> dict:
    """Load and validate one export or a JSON bundle of several exports.

    Returns per-factor windows, exclusivity rules, protocol provenance and
    the SHA-256 of the exact bytes parsed.
    """
    raw = Path(path).read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: constraints are not valid JSON: {exc}") from exc
    exports = _as_exports(payload)
    if not exports:
        raise ValueError(f"{path}: constraints bundle is empty")
    factors: dict[str, dict] = {}
    exclusivity: list[dict] = []
    protocols: list[dict] = []
    for export in exports:
        if not isinstance(export, dict):
            raise ValueError(f"{path}: each constraints entry must be an object")
        version = export.get("schema_version")
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"{path}: unsupported constraints schema_version {version!r}")
        protocol_id = export.get("protocol_id")
        if not isinstance(protocol_id, str) or not protocol_id.strip():
            raise ValueError(f"{path}: constraints entry is missing protocol_id")
        parameters = export.get("parameters")
        if not parameters:
            raise ValueError(f"{path}: {protocol_id} exports no parameters")
        protocols.append({
            "protocol_id": protocol_id,
            "protocol_sha256": export.get("protocol_sha256"),
        })
        for parameter in parameters:
            name = parameter.get("name")
            low, high = parameter.get("low"), parameter.get("high")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"{path}: parameter name must be a nonblank string")
            if name in factors:
                raise ValueError(f"{path}: duplicate constrained factor {name}")
            if not _number(low) or not _number(high) or not low <= high:
                raise ValueError(
                    f"{path}: {name} window [{low}, {high}] is not a finite ordered interval"
                )
            required = parameter.get("required", False)
            if not isinstance(required, bool):
                raise ValueError(f"{path}: {name} required flag must be boolean")
            factors[name] = {
                "low": float(low),
                "high": float(high),
                "unit": parameter.get("unit", ""),
                "note": parameter.get("note", ""),
                "protocol_id": protocol_id,
                "protocol_sha256": export.get("protocol_sha256"),
                "required": required,
            }
        for rule in export.get("exclusivity", []):
            if not isinstance(rule, dict):
                raise ValueError(f"{path}: exclusivity rules must be objects")
            members = rule.get("parameters")
            if (not isinstance(members, list) or len(members) < 2
                    or any(name not in factors for name in members)):
                raise ValueError(
                    f"{path}: exclusivity rule must name at least two known parameters"
                )
            if any(not factors[name]["required"] for name in members):
                raise ValueError(
                    f"{path}: exclusivity members must be required parameters "
                    "so an omitted column cannot bypass the rule"
                )
            if rule.get("rule") != AT_MOST_ONE_POSITIVE:
                raise ValueError(f"{path}: unsupported exclusivity rule {rule.get('rule')!r}")
            minimum_positive = rule.get("minimum_positive", 0)
            if (isinstance(minimum_positive, bool) or not isinstance(minimum_positive, int)
                    or minimum_positive not in (0, 1)):
                raise ValueError(f"{path}: minimum_positive must be 0 or 1")
            rule_inactive = rule.get("inactive_values", {})
            if not isinstance(rule_inactive, dict) or any(
                member not in members or not _number(value) or value > 0
                for member, value in rule_inactive.items()
            ):
                raise ValueError(f"{path}: inactive_values must map rule members to finite nonpositive numbers")
            for name, value in rule_inactive.items():
                factors[name]["inactive_value"] = float(value)
            exclusivity.append({
                "parameters": list(members),
                "rule": AT_MOST_ONE_POSITIVE,
                "minimum_positive": minimum_positive,
                "rationale": rule.get("rationale", ""),
                "inactive_values": {name: float(value) for name, value in rule_inactive.items()},
            })
    return {
        "factors": factors,
        "exclusivity": exclusivity,
        "protocols": protocols,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
    }
