"""Dataset-role, manifest, and freeze validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


VALID_ROLES = {"fit", "validation", "calibration", "locked_test", "external_test"}
VALID_PROTOCOL_TYPES = {"metric_radius", "official_evaluator", "frame_correspondence", "pose_rule"}


class ProtocolError(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def validate_dataset_manifest(manifest: dict, allow_template: bool = False) -> None:
    if manifest.get("format") != "mpef_dataset_manifest_v1":
        raise ProtocolError("unsupported dataset manifest format")
    records = manifest.get("datasets")
    if not isinstance(records, list) or not records:
        raise ProtocolError("dataset manifest must contain a non-empty datasets list")
    seen_ids: set[str] = set()
    role_hashes: dict[str, set[str]] = {role: set() for role in VALID_ROLES}
    for record in records:
        dataset_id = record.get("dataset_id")
        role = record.get("role")
        if not dataset_id or dataset_id in seen_ids:
            raise ProtocolError(f"dataset_id must be present and unique: {dataset_id!r}")
        seen_ids.add(dataset_id)
        if role not in VALID_ROLES:
            raise ProtocolError(f"invalid role for {dataset_id}: {role!r}")
        protocol = record.get("primary_protocol", {})
        if protocol.get("type") not in VALID_PROTOCOL_TYPES:
            raise ProtocolError(f"invalid primary protocol for {dataset_id}")
        if protocol.get("type") == "metric_radius":
            radius = protocol.get("radius_m")
            if not isinstance(radius, (int, float)) or radius <= 0:
                raise ProtocolError(f"metric protocol for {dataset_id} requires radius_m>0")
        if not record.get("protocol_id"):
            raise ProtocolError(f"{dataset_id} requires a typed protocol_id")
        if record.get("query_manifest_sha256") == "TO_BE_FROZEN" and not allow_template:
            raise ProtocolError(f"{dataset_id} query manifest is not frozen")
        for digest in record.get("image_sha256", []):
            if digest in role_hashes[role]:
                raise ProtocolError(f"duplicate image hash within role {role}: {digest}")
            role_hashes[role].add(digest)

    protected = role_hashes["locked_test"] | role_hashes["external_test"]
    training = role_hashes["fit"] | role_hashes["validation"] | role_hashes["calibration"]
    overlap = protected & training
    if overlap:
        raise ProtocolError(f"train/protected image-hash overlap: {sorted(overlap)[:3]}")


def validate_method_freeze(freeze: dict, allow_template: bool = False) -> None:
    if freeze.get("format") != "mpef_method_freeze_v1":
        raise ProtocolError("unsupported method freeze format")
    if freeze.get("fixed_top_k") != 100:
        raise ProtocolError("primary MPEF protocol requires fixed_top_k=100")
    if freeze.get("query_dependent_matching") is not False:
        raise ProtocolError("query-dependent matching is prohibited")
    if freeze.get("old_result_reuse") is not False:
        raise ProtocolError("old result reuse must be false")
    if not allow_template and freeze.get("status") != "frozen":
        raise ProtocolError("method freeze status must be 'frozen'")
    if not allow_template and freeze.get("source_commit") in {None, "TO_BE_FROZEN"}:
        raise ProtocolError("source_commit must be frozen")


def validate_hypotheses_freeze(freeze: dict, allow_template: bool = False) -> None:
    if freeze.get("format") != "mpef_hypotheses_freeze_v1":
        raise ProtocolError("unsupported hypotheses freeze format")
    hypotheses = freeze.get("co_primary_hypotheses", [])
    if [h.get("endpoint") for h in hypotheses] != ["recall_at_1", "candidate_log_loss"]:
        raise ProtocolError("co-primary endpoints must be Recall@1 and candidate log loss")
    if freeze.get("multiplicity") != "holm_across_two_co_primary":
        raise ProtocolError("co-primary hypotheses require Holm correction")
    if not allow_template and freeze.get("status") != "frozen":
        raise ProtocolError("hypotheses freeze status must be 'frozen'")


def select_primary_protocol_rows(rows: Iterable[dict], protocol_by_dataset: dict[str, str]) -> list[dict]:
    """Select exactly one typed primary-protocol row per dataset/query/method.

    This intentionally avoids numeric threshold comparisons and NaN sentinels.
    """

    selected = [
        row for row in rows
        if row.get("protocol_id") == protocol_by_dataset.get(row.get("dataset_id"))
    ]
    seen: set[tuple] = set()
    for row in selected:
        key = (row.get("dataset_id"), row.get("query_id"), row.get("method_id"))
        if key in seen:
            raise ProtocolError(f"duplicate primary-protocol row: {key}")
        seen.add(key)
    return selected

