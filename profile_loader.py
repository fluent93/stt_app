"""profiles.yaml 로드. transcribe_korean 에 넣을 실행 파라미터만 반환합니다."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_ALLOWED_KEYS = frozenset(
    {
        "model_size",
        "beam_size",
        "cpu_threads",
        "split_mode",
        "overlap_seconds",
        "compute_type",
    }
)


def profiles_yaml_path() -> Path:
    return Path(__file__).resolve().parent / "profiles.yaml"


def load_profiles(path: Path | None = None) -> dict[str, dict[str, Any]]:
    p = path or profiles_yaml_path()
    if not p.is_file():
        raise FileNotFoundError(f"profiles.yaml 을 찾을 수 없습니다: {p}")
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("profiles.yaml 최상위는 프로필 이름 → 설정 dict 여야 합니다.")
    out: dict[str, dict[str, Any]] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            continue
        cleaned = {k: spec[k] for k in _ALLOWED_KEYS if k in spec}
        out[str(name)] = cleaned
    if not out:
        raise ValueError("profiles.yaml 에 유효한 프로필이 없습니다.")
    return out


def profile_exec_kw(profile: dict[str, Any]) -> dict[str, Any]:
    """transcribe_korean 에 그대로 전달 가능한 dict (허용 키만)."""
    return {k: profile[k] for k in _ALLOWED_KEYS if k in profile}
