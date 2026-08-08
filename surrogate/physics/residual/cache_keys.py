from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


SOURCE_AWARE_CACHE_KEY_VERSION = 2

_CASE_SUFFIX_PATTERN = re.compile(r"^(?P<prefix>.+?)_case_\d+_000_vol$")
_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def normalize_optional_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def normalize_source_info(source_info: Optional[Mapping[str, Any]]) -> dict[str, str]:
    payload = dict(source_info or {})
    return {
        "source_name": normalize_optional_text(payload.get("source_name")),
        "source_kind": normalize_optional_text(payload.get("source_kind")),
        "source_chunk": normalize_optional_text(payload.get("source_chunk")),
        "source_index_path": normalize_optional_text(payload.get("source_index_path")),
        "source_shard_root": normalize_optional_text(payload.get("source_shard_root")),
        "source_shard_path": normalize_optional_text(payload.get("source_shard_path")),
    }


def extract_source_info_from_sample(sample: Mapping[str, Any]) -> dict[str, str]:
    embedded = sample.get("source_info")
    if isinstance(embedded, Mapping):
        return normalize_source_info(embedded)
    return normalize_source_info(
        {
            "source_name": sample.get("source_name"),
            "source_kind": sample.get("source_kind"),
            "source_chunk": sample.get("source_chunk"),
            "source_index_path": sample.get("source_index_path"),
            "source_shard_root": sample.get("source_shard_root"),
            "source_shard_path": sample.get("source_shard_path"),
        }
    )


def collapse_case_suffix(cgns_basename: str) -> str:
    basename = Path(str(cgns_basename)).name
    stem = basename[:-5] if basename.endswith(".cgns") else basename
    match = _CASE_SUFFIX_PATTERN.match(stem)
    return match.group("prefix") if match else stem


def sanitize_cache_prefix(text: str) -> str:
    prefix = _SANITIZE_PATTERN.sub("_", str(text)).strip("._-")
    if not prefix:
        raise ValueError(f"Could not derive cache prefix from {text!r}")
    return prefix


def sanitize_cache_stem(cgns_basename: str) -> str:
    return sanitize_cache_prefix(collapse_case_suffix(cgns_basename))


def _round_float(value: Any) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        return scalar
    return round(scalar, 9)


def _canonicalize_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize_jsonable(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_jsonable(item) for item in value]
    if isinstance(value, Path):
        return normalize_optional_text(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return _round_float(value)
    if value is None:
        return None
    return normalize_optional_text(value)


def build_source_aware_cache_identity(
    *,
    cache_family: str,
    cgns_name: str,
    source_info: Optional[Mapping[str, Any]] = None,
    dataset_index: Optional[int] = None,
    flow_conditions: Optional[Mapping[str, Any]] = None,
    resolved_path: Optional[str | Path] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "cache_key_version": int(SOURCE_AWARE_CACHE_KEY_VERSION),
        "cache_family": str(cache_family),
        "cgns_name": str(cgns_name),
        "source_info": normalize_source_info(source_info),
    }
    if dataset_index is not None:
        identity["dataset_index"] = int(dataset_index)
    resolved_path_text = normalize_optional_text(resolved_path)
    if resolved_path_text:
        identity["resolved_path"] = resolved_path_text
    if flow_conditions:
        identity["flow_conditions"] = {
            str(key): _round_float(value)
            for key, value in sorted(flow_conditions.items(), key=lambda item: str(item[0]))
        }
    if extra:
        identity["extra"] = _canonicalize_jsonable(extra)
    return identity


def build_source_aware_cache_key(
    *,
    prefix_text: str,
    identity: Mapping[str, Any],
    digest_len: int = 16,
) -> str:
    prefix = sanitize_cache_prefix(prefix_text)
    key_text = json.dumps(
        _canonicalize_jsonable(dict(identity)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha1(key_text.encode("utf-8")).hexdigest()[: int(digest_len)]
    return f"{prefix}__{digest}"


def build_hashed_cache_key(cgns_basename: str, identity: Mapping[str, Any], *, digest_len: int = 16) -> str:
    return build_source_aware_cache_key(
        prefix_text=Path(str(cgns_basename)).stem,
        identity=identity,
        digest_len=digest_len,
    )


def build_cgns_case_cache_identity(
    cgns_basename: str,
    *,
    cache_namespace: str,
    source_info: Optional[Mapping[str, Any]] = None,
    dataset_index: Optional[int] = None,
    flow_conditions: Optional[Mapping[str, Any]] = None,
    resolved_path: Optional[str | Path] = None,
    expected_shape: Optional[Sequence[int]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    merged_extra = dict(extra or {})
    shape_list = shape_to_list(expected_shape)
    if shape_list is not None:
        merged_extra["expected_shape"] = shape_list
    geometry_name = collapse_case_suffix(cgns_basename)
    return build_source_aware_cache_identity(
        cache_family=str(cache_namespace),
        cgns_name=geometry_name,
        source_info=source_info,
        dataset_index=dataset_index,
        flow_conditions=flow_conditions,
        resolved_path=resolved_path,
        extra=merged_extra or None,
    )


def cache_identity_matches(meta_payload: Mapping[str, Any], identity: Mapping[str, Any]) -> bool:
    meta_identity = meta_payload.get("cache_identity")
    if meta_identity is None:
        return False
    return _canonicalize_jsonable(meta_identity) == _canonicalize_jsonable(dict(identity))


def shape_to_list(expected_shape: Optional[Sequence[int]]) -> list[int] | None:
    if expected_shape is None:
        return None
    return [int(dim) for dim in expected_shape]
