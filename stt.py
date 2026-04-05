from __future__ import annotations

import logging
import os
import re
import time
import wave
from typing import Any

import numpy as np
import streamlit as st
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, collect_chunks, get_speech_timestamps

from utils import (
    load_mono_16k_wav_as_float32,
    split_mono_float32_into_segments,
)

# 병렬 chunk STT: WhisperModel은 스레드 안전이 보장되지 않으며, condition_on_previous_text로
# 청크 간 문맥을 넘기는 현재 구조에서는 순차 의존이 있어 단순 ThreadPool 적용이 어렵다.
# 순서를 유지한 병렬화를 하려면 (1) 청크별 WhisperModel 인스턴스(메모리 N배) 또는
# (2) 문맥 없이 독립 transcribe 후 후처리 병합(품질 변동) 같은 트레이드오프가 필요하다.

logger = logging.getLogger(__name__)

# Whisper initial_prompt는 토큰 상한(~224)이 있어, 이전 청크 문맥은 꼬리만 넘깁니다.
_CHUNK_CONTEXT_MAX_CHARS = 448

# STT 기본 튜닝(환경변수 미설정 시). M3 등 CPU 전용 시 속도 우선 프리셋.
_STT_DEFAULTS: dict[str, str | int] = {
    "model_size": "small",
    "beam_size": 1,
    "cpu_threads": 4,
}


def _env_model_size() -> str:
    raw = os.environ.get("WHISPER_MODEL_SIZE", _STT_DEFAULTS["model_size"])
    return (str(raw).strip() or str(_STT_DEFAULTS["model_size"]))


def _env_beam_size() -> int:
    raw = os.environ.get("WHISPER_BEAM_SIZE", str(_STT_DEFAULTS["beam_size"])).strip()
    try:
        b = int(raw)
    except ValueError:
        b = int(_STT_DEFAULTS["beam_size"])
    return max(1, min(b, 10))


def _env_cpu_threads() -> int:
    raw = os.environ.get("WHISPER_CPU_THREADS", str(_STT_DEFAULTS["cpu_threads"])).strip()
    try:
        n = int(raw)
    except ValueError:
        n = int(_STT_DEFAULTS["cpu_threads"])
    return max(1, min(n, 32))


def default_cpu_threads_for_ui() -> int:
    """앱 UI 슬라이더 기본값(환경변수·코드 기본 반영)."""
    return _env_cpu_threads()


def _env_compute_type() -> str:
    raw = os.environ.get("WHISPER_COMPUTE_TYPE", "int8").strip()
    return raw or "int8"


def _faster_whisper_runtime_settings(
    model_size_key: str,
    cpu_threads: int,
    compute_type: str | None = None,
) -> dict[str, Any]:
    """
    WhisperModel()에 넣는 값과 동일. 로그/UI에서 실제 실행 설정 확인용.
    """
    name = (model_size_key or "").strip() or str(_STT_DEFAULTS["model_size"])
    cpu_threads = max(1, min(int(cpu_threads), 32))
    ct = (compute_type or "").strip() or _env_compute_type()
    return {
        "inference_model_name": name,
        "inference_device": "cpu",
        "inference_compute_type": ct,
        "inference_cpu_threads": cpu_threads,
        "inference_num_workers": 1,
    }


def _augment_runtime_from_loaded_model(model: WhisperModel, rt: dict[str, Any]) -> dict[str, Any]:
    """가능하면 ctranslate2 쪽에 보고되는 device 등을 덧붙입니다."""
    out = dict(rt)
    try:
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "device"):
            out["ctranslate2_device"] = str(inner.device)
    except Exception:
        pass
    return out


def _tail_for_initial_prompt(text: str, max_chars: int = _CHUNK_CONTEXT_MAX_CHARS) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    return t[-max_chars:] if len(t) > max_chars else t


def _collapse_ws_for_overlap(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _strip_leading_content_chars(raw: str, n: int) -> str:
    """공백은 건너뛰며 앞에서부터 비공백 문자 n개 분량을 제거."""
    if n <= 0:
        return raw
    seen = 0
    i = 0
    while i < len(raw) and seen < n:
        if not raw[i].isspace():
            seen += 1
        i += 1
    while i < len(raw) and raw[i].isspace():
        i += 1
    return raw[i:]


def _strip_first_k_words(raw: str, k: int) -> str:
    if k <= 0:
        return raw
    words = list(re.finditer(r"\S+", raw))
    if len(words) < k:
        return raw
    return raw[words[k - 1].end() :].lstrip()


def _longest_suffix_prefix_overlap_chars(prev: str, nxt: str, min_len: int, max_len: int) -> int:
    pc = _collapse_ws_for_overlap(prev)
    nc = _collapse_ws_for_overlap(nxt)
    if not pc or not nc:
        return 0
    upper = min(len(pc), len(nc), max_len)
    if upper < min_len:
        return 0
    for length in range(upper, min_len - 1, -1):
        if pc[-length:] == nc[:length]:
            return length
    return 0


def _write_mono_wav_f32(path: str, audio: np.ndarray, sample_rate: int = 16000) -> None:
    """float32 [-1, 1] 모노 PCM을 16-bit WAV로 저장."""
    pcm = np.clip(audio.astype(np.float32) * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _vad_options_from_env() -> VadOptions:
    """faster-whisper transcribe의 vad_parameters와 맞추기 쉽게 환경변수로 조정."""
    return VadOptions(
        threshold=float(os.environ.get("WHISPER_VAD_THRESHOLD", "0.5")),
        min_speech_duration_ms=int(os.environ.get("WHISPER_VAD_MIN_SPEECH_MS", "0")),
        max_speech_duration_s=float(os.environ.get("WHISPER_VAD_MAX_SPEECH_S", "30")),
        min_silence_duration_ms=int(os.environ.get("WHISPER_VAD_MIN_SILENCE_MS", "500")),
        speech_pad_ms=int(os.environ.get("WHISPER_VAD_SPEECH_PAD_MS", "200")),
    )


def _chunk_durations_seconds(chunks: list[np.ndarray], sample_rate: int) -> list[float]:
    """각 청크 길이(초). 빈 배열은 제외."""
    out: list[float] = []
    for c in chunks:
        if c is None or np.size(c) == 0:
            continue
        out.append(float(np.size(c)) / float(sample_rate))
    return out


def _log_vad_chunk_distribution(stage: str, chunks: list[np.ndarray], sample_rate: int) -> None:
    """VAD 단계별 청크 길이(초) 분포를 INFO로 남깁니다. Streamlit은 터미널 로그에서 확인합니다."""
    durs = _chunk_durations_seconds(chunks, sample_rate)
    if not durs:
        logger.info("VAD chunk lengths [%s]: (no chunks)", stage)
        return
    arr = np.array(durs, dtype=np.float64)
    if len(durs) <= 60:
        parts = ", ".join(f"{x:.3f}" for x in durs)
    else:
        head = ", ".join(f"{x:.3f}" for x in durs[:25])
        tail = ", ".join(f"{x:.3f}" for x in durs[-15:])
        parts = f"{head}, …({len(durs) - 40} omitted)…, {tail}"
    logger.info(
        "VAD chunk lengths [%s]: count=%d each_s=[%s] min=%.3f max=%.3f mean=%.3f median=%.3f",
        stage,
        len(durs),
        parts,
        float(arr.min()),
        float(arr.max()),
        float(arr.mean()),
        float(np.median(arr)),
    )


def _backward_merge_trailing_for_target(
    packed: list[np.ndarray],
    sample_rate: int,
    target_seconds: float,
    max_seconds: float,
) -> list[np.ndarray]:
    """마지막 청크만 target 미만이면, max_seconds 이내면 앞 청크에 붙입니다."""
    def dur(a: np.ndarray) -> float:
        return float(a.size) / float(sample_rate)

    changed = True
    while changed and len(packed) >= 2:
        changed = False
        if dur(packed[-1]) < target_seconds - 1e-9 and dur(packed[-2]) + dur(packed[-1]) <= max_seconds + 1e-6:
            packed[-2] = np.concatenate([packed[-2], packed[-1]])
            packed.pop()
            changed = True
    return packed


def _merge_vad_chunks_target_length(
    chunks: list[np.ndarray],
    sample_rate: int,
    target_seconds: float,
    max_seconds: float,
) -> list[np.ndarray]:
    """
    시간 순 인접 청크를 붙여, 가능한 한 각 조각이 target_seconds에 가깝도록 만듭니다.
    단, 합쳐진 길이는 max_seconds를 넘지 않습니다 (Whisper·메모리 상한).

    merge_min 방식과 달리 10~20초짜리 인접 청크도 합쳐 chunk 수를 줄입니다.
    """
    ch: list[np.ndarray] = [
        np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks if c is not None and np.size(c) > 0
    ]
    if not ch:
        return []
    max_seconds = max(float(max_seconds), 5.0)
    target_seconds = float(target_seconds)
    target_seconds = min(max(target_seconds, 5.0), max_seconds)

    def dur(a: np.ndarray) -> float:
        return float(a.size) / float(sample_rate)

    out: list[np.ndarray] = []
    acc = ch[0]
    for nxt in ch[1:]:
        if dur(acc) < target_seconds - 1e-9 and dur(acc) + dur(nxt) <= max_seconds + 1e-6:
            acc = np.concatenate([acc, nxt])
        else:
            out.append(acc)
            acc = nxt
    out.append(acc)
    return _backward_merge_trailing_for_target(out, sample_rate, target_seconds, max_seconds)


def _merge_short_vad_chunks(
    chunks: list[np.ndarray],
    sample_rate: int,
    min_seconds: float,
    max_seconds: float,
) -> list[np.ndarray]:
    """
    VAD collect_chunks 결과에서 길이가 min_seconds 미만인 조각을 시간 순 이웃과 병합합니다.
    max_seconds를 넘기지 않는 한에서만 병합해, Whisper 입력 길이 상한을 유지합니다.
    우선 뒤쪽(다음 청크)과 붙여 문맥이 앞에서 이어지도록 합니다.
    """
    if not chunks or min_seconds <= 0:
        return chunks
    ch: list[np.ndarray] = [
        np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks if c is not None and np.size(c) > 0
    ]
    if len(ch) <= 1:
        return ch

    max_seconds = max(max_seconds, min_seconds)

    def dur(arr: np.ndarray) -> float:
        return float(arr.size) / float(sample_rate)

    changed = True
    while changed and len(ch) > 1:
        changed = False
        for i in range(len(ch)):
            if dur(ch[i]) >= min_seconds - 1e-9:
                continue

            can_r = (
                i + 1 < len(ch)
                and dur(ch[i]) + dur(ch[i + 1]) <= max_seconds + 1e-6
            )
            can_l = i > 0 and dur(ch[i - 1]) + dur(ch[i]) <= max_seconds + 1e-6

            if can_r:
                ch[i] = np.concatenate([ch[i], ch[i + 1]])
                del ch[i + 1]
                changed = True
                break
            if can_l:
                ch[i - 1] = np.concatenate([ch[i - 1], ch[i]])
                del ch[i]
                changed = True
                break

    return ch


def _resolve_vad_chunk_bounds(
    max_chunk_seconds: float | None,
    merge_min_seconds: float | None,
    target_chunk_seconds: float | None,
) -> tuple[float, float, float]:
    """
    max_chunk_s: collect_chunks·패킹 공통 상한(초).
    merge_min_s: 레거시(짧은 조각만 붙이기)용; WHISPER_VAD_USE_SHORT_MERGE=1 일 때만 사용.
    target_chunk_s: 목표 길이(초). 기본 24. 0 이하이면 target 패킹 생략하고 short-merge만.
    """
    if max_chunk_seconds is not None:
        max_c = float(max_chunk_seconds)
    else:
        max_c = float(os.environ.get("WHISPER_VAD_MAX_CHUNK_SECONDS", "30"))
    if max_c < 5.0:
        max_c = 5.0

    if merge_min_seconds is not None:
        merge_m = float(merge_min_seconds)
    else:
        merge_m = float(os.environ.get("WHISPER_VAD_MERGE_MIN_SECONDS", "10"))
    if merge_m < 3.0:
        merge_m = 3.0

    if target_chunk_seconds is not None:
        target = float(target_chunk_seconds)
    else:
        target = float(os.environ.get("WHISPER_VAD_TARGET_CHUNK_SECONDS", "24"))
    if target < 0:
        target = 0.0
    elif 0 < target < 5.0:
        target = 5.0
    if target > max_c:
        target = max_c
    return max_c, merge_m, target


def _vad_pack_audio_to_arrays(
    audio: np.ndarray,
    *,
    max_chunk_seconds: float | None = None,
    merge_min_seconds: float | None = None,
    target_chunk_seconds: float | None = None,
) -> tuple[list[np.ndarray], float, float, float, int, int]:
    """
    mono float32 오디오에 대해 VAD·collect_chunks·target 패킹까지 수행해 numpy 청크 목록만 반환합니다.
    디스크에 세그먼트 WAV를 쓰지 않습니다.
    반환: (배열 목록, max_chunk_s, merge_min_s, target_chunk_s, n_after_collect, n_after_pack)
    """
    max_chunk_s, merge_min_s, target_chunk_s = _resolve_vad_chunk_bounds(
        max_chunk_seconds, merge_min_seconds, target_chunk_seconds
    )
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    full_contig = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.size == 0:
        return [], max_chunk_s, merge_min_s, target_chunk_s, 0, 0

    vad_opt = _vad_options_from_env()
    timestamps = get_speech_timestamps(audio, vad_options=vad_opt)
    if not timestamps:
        return [full_contig], max_chunk_s, merge_min_s, target_chunk_s, 0, 0

    audio_chunks, _meta = collect_chunks(
        audio,
        timestamps,
        sampling_rate=16000,
        max_duration=max_chunk_s,
    )

    raw_list = [np.asarray(x, dtype=np.float32).reshape(-1) for x in audio_chunks if x is not None and np.size(x) > 0]
    n_collect = len(raw_list)
    _log_vad_chunk_distribution("after_collect_chunks", raw_list, 16000)

    use_short_only = os.environ.get("WHISPER_VAD_USE_SHORT_MERGE", "0").lower() in ("1", "true", "yes")
    if use_short_only or target_chunk_s <= 1e-9:
        packed = _merge_short_vad_chunks(raw_list, 16000, merge_min_s, max_chunk_s)
        pack_mode = "short_merge_min"
    else:
        packed = _merge_vad_chunks_target_length(raw_list, 16000, target_chunk_s, max_chunk_s)
        pack_mode = "target_length"
        if merge_min_s <= max_chunk_s + 1e-9:
            packed = _merge_short_vad_chunks(packed, 16000, merge_min_s, max_chunk_s)

    n_pack = len(packed)
    _log_vad_chunk_distribution(f"after_pack({pack_mode})", packed, 16000)
    logger.info(
        "VAD chunk counts: after_collect=%d after_pack=%d (target_s=%.3f max_s=%.3f mode=%s)",
        n_collect,
        n_pack,
        target_chunk_s,
        max_chunk_s,
        pack_mode,
    )

    min_samples = int(16000 * float(os.environ.get("WHISPER_MIN_CHUNK_SECONDS", "0.3")))
    out_arrays: list[np.ndarray] = []
    for chunk_arr in packed:
        if chunk_arr.size < min_samples:
            continue
        out_arrays.append(np.ascontiguousarray(chunk_arr, dtype=np.float32))

    if not out_arrays:
        return [full_contig], max_chunk_s, merge_min_s, target_chunk_s, n_collect, n_pack
    return out_arrays, max_chunk_s, merge_min_s, target_chunk_s, n_collect, n_pack


def _dedupe_overlap_next_chunk(prev_raw_chunk: str, next_raw_chunk: str) -> str:
    """
    이전 청크(raw 인식) 끝과 다음 청크(raw) 앞의 겹침을 제거한 뒤 텍스트만 반환.
    공백 무시 문자 일치 후, 실패 시 단어 접두 일치로 보조.
    """
    prev_raw_chunk = (prev_raw_chunk or "").strip()
    next_raw_chunk = (next_raw_chunk or "").strip()
    if not prev_raw_chunk:
        return next_raw_chunk
    if not next_raw_chunk:
        return ""

    min_chars = int(os.environ.get("WHISPER_OVERLAP_DEDUP_MIN_CHARS", "12"))
    max_chars = int(os.environ.get("WHISPER_OVERLAP_DEDUP_MAX_CHARS", "2500"))
    L = _longest_suffix_prefix_overlap_chars(prev_raw_chunk, next_raw_chunk, min_chars, max_chars)
    if L > 0:
        return _strip_leading_content_chars(next_raw_chunk, L).strip()

    min_words = int(os.environ.get("WHISPER_OVERLAP_DEDUP_MIN_WORDS", "3"))
    pw = prev_raw_chunk.split()
    nw = next_raw_chunk.split()
    max_k = min(len(pw), len(nw))
    for k in range(max_k, min_words - 1, -1):
        if pw[-k:] == nw[:k]:
            return _strip_first_k_words(next_raw_chunk, k).strip()
    return next_raw_chunk


def _is_ssl_certificate_error(message: str) -> bool:
    msg = (message or "").lower()
    return (
        "certificate_verify_failed" in msg
        or "certificate verify failed" in msg
        or "self signed certificate" in msg
        or "ssl" in msg and "certificate" in msg
    )


def _disable_huggingface_ssl_verification() -> None:
    """
    Hugging Face Hub 다운로드 시 SSL 검증을 끄기 위한 임시 우회.

    환경변수로도 안 먹는 환경이 있어, fast-whisper가 호출하는
    huggingface_hub의 httpx client factory를 직접 바꿉니다.
    """
    try:
        import httpx
        import huggingface_hub.utils._http as hf_http

        def client_factory() -> httpx.Client:
            return httpx.Client(
                event_hooks={"request": [hf_http.hf_request_event_hook]},
                follow_redirects=True,
                timeout=None,
                verify=False,
            )

        def async_client_factory() -> httpx.AsyncClient:
            return httpx.AsyncClient(
                event_hooks={"request": [hf_http.hf_request_event_hook], "response": [hf_http.async_hf_response_event_hook]},
                follow_redirects=True,
                timeout=None,
                verify=False,
            )

        hf_http.set_client_factory(client_factory)
        hf_http.set_async_client_factory(async_client_factory)
    except Exception:
        # 우회 설정이 실패하면 원래 예외를 그대로 처리합니다.
        pass


@st.cache_resource(show_spinner=False)
def _load_model(model_size: str, cpu_threads: int, compute_type: str | None = None) -> WhisperModel:
    """
    faster-whisper 모델을 (model_size, cpu_threads, compute_type) 조합별로 캐시해 재사용합니다.
    """
    rt = _faster_whisper_runtime_settings(model_size, cpu_threads, compute_type)
    download_root = os.environ.get("WHISPER_MODEL_DIR") or None
    local_files_only = os.environ.get("WHISPER_LOCAL_FILES_ONLY", "0").lower() in (
        "1",
        "true",
        "yes",
    )

    return WhisperModel(
        rt["inference_model_name"],
        device=rt["inference_device"],
        cpu_threads=rt["inference_cpu_threads"],
        num_workers=rt["inference_num_workers"],
        compute_type=rt["inference_compute_type"],
        download_root=download_root,
        local_files_only=local_files_only,
    )


def _env_split_settings() -> tuple[str, int, float]:
    """WHISPER_SPLIT_MODE (vad|fixed|full), 고정 분할용 segment/overlap 초 단위."""
    split_mode = os.environ.get("WHISPER_SPLIT_MODE", "vad").strip().lower()
    if split_mode not in ("vad", "fixed", "full"):
        split_mode = "vad"
    segment_seconds = int(os.environ.get("WHISPER_SEGMENT_SECONDS", "10"))
    if segment_seconds < 5:
        segment_seconds = 5
    overlap_seconds = float(os.environ.get("WHISPER_OVERLAP_SECONDS", "0"))
    if overlap_seconds < 0:
        overlap_seconds = 0.0
    return split_mode, segment_seconds, overlap_seconds


def transcribe_korean(
    wav_path: str,
    progress_callback=None,
    *,
    model_size: str | None = None,
    beam_size: int | None = None,
    overlap_seconds: float | None = None,
    cpu_threads: int | None = None,
    vad_max_chunk_seconds: float | None = None,
    vad_merge_min_seconds: float | None = None,
    vad_target_chunk_seconds: float | None = None,
    split_mode: str | None = None,
    compute_type: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    wav_path(16kHz, mono WAV 가정)로부터 한국어 STT 수행.
    model_size / beam_size / overlap_seconds / cpu_threads / compute_type / vad_* / split_mode 를 주면 해당 실행에만 적용(환경변수보다 우선).
    vad_* 는 split_mode=vad 일 때만 사용됩니다.
    반환: (텍스트, 메트릭 dict).
    """
    t_stt_wall = time.perf_counter()
    env_split_mode, segment_seconds, overlap_env = _env_split_settings()
    if split_mode is not None:
        sm = str(split_mode).strip().lower()
        resolved_split_mode = sm if sm in ("vad", "fixed", "full") else env_split_mode
    else:
        resolved_split_mode = env_split_mode
    resolved_model = (model_size or "").strip() or _env_model_size()
    resolved_beam = max(1, min(int(beam_size) if beam_size is not None else _env_beam_size(), 10))
    resolved_overlap = float(overlap_seconds) if overlap_seconds is not None else overlap_env
    if resolved_overlap < 0:
        resolved_overlap = 0.0
    resolved_cpu_threads = max(1, min(int(cpu_threads) if cpu_threads is not None else _env_cpu_threads(), 32))
    if compute_type is not None:
        _ct = str(compute_type).strip()
        resolved_compute_type = _ct if _ct else _env_compute_type()
    else:
        resolved_compute_type = _env_compute_type()

    def _finalize(text: str, inner: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        inner = dict(inner)
        inner["stt_total_seconds"] = time.perf_counter() - t_stt_wall
        n = int(inner.get("chunk_count") or 0)
        tr = float(inner.get("transcribe_seconds") or 0.0)
        inner["avg_chunk_seconds"] = (tr / n) if n else 0.0
        logger.info(
            "STT 완료 wall=%.3fs transcribe=%.3fs prep=%.3fs chunks=%d avg_chunk=%.3fs "
            "model=%s beam=%d overlap=%.2f split=%s vad_max_chunk=%s vad_target=%s vad_merge_min=%s "
            "vad_collect=%s vad_pack=%s | "
            "fw_device=%s fw_compute=%s fw_cpu_threads=%s fw_model_name=%s ctranslate2_device=%s",
            inner["stt_total_seconds"],
            tr,
            float(inner.get("prep_seconds") or 0.0),
            n,
            inner["avg_chunk_seconds"],
            inner.get("model_size", ""),
            int(inner.get("beam_size") or 0),
            float(inner.get("overlap_seconds") or 0.0),
            inner.get("split_mode", ""),
            inner.get("vad_max_chunk_seconds"),
            inner.get("vad_target_chunk_seconds"),
            inner.get("vad_merge_min_seconds"),
            inner.get("vad_collect_chunk_count"),
            inner.get("vad_pack_chunk_count"),
            inner.get("inference_device", ""),
            inner.get("inference_compute_type", ""),
            inner.get("inference_cpu_threads", ""),
            inner.get("inference_model_name", ""),
            inner.get("ctranslate2_device", "n/a"),
        )
        return text, inner

    def _merge_fw_runtime(
        model: WhisperModel,
        inner: dict[str, Any],
        name_for_settings: str,
        threads: int,
        fw_compute_type: str | None = None,
    ) -> dict[str, Any]:
        inner = dict(inner)
        rt = _augment_runtime_from_loaded_model(
            model,
            _faster_whisper_runtime_settings(name_for_settings, threads, fw_compute_type),
        )
        inner.update(rt)
        return inner

    try:
        model = _load_model(resolved_model, resolved_cpu_threads, resolved_compute_type)

        text, inner = _transcribe_with_segments(
            model=model,
            wav_path=wav_path,
            segment_seconds=segment_seconds,
            overlap_seconds=resolved_overlap,
            split_mode=resolved_split_mode,
            model_size=resolved_model,
            beam_size=resolved_beam,
            progress_callback=progress_callback,
            vad_max_chunk_seconds=vad_max_chunk_seconds,
            vad_merge_min_seconds=vad_merge_min_seconds,
            vad_target_chunk_seconds=vad_target_chunk_seconds,
        )
        inner = _merge_fw_runtime(
            model, inner, resolved_model, resolved_cpu_threads, resolved_compute_type
        )
        return _finalize(text, inner)
    except Exception as e:
        msg = str(e)
        if _is_ssl_certificate_error(msg):
            # 임시 우회 후 1회 재시도
            _disable_huggingface_ssl_verification()
            try:
                # 캐시된 모델이 남아있으면 재시도에 방해될 수 있어 제거 시도
                if hasattr(_load_model, "clear"):
                    _load_model.clear()
            except Exception:
                pass

            model = _load_model(resolved_model, resolved_cpu_threads, resolved_compute_type)
            text, inner = _transcribe_with_segments(
                model=model,
                wav_path=wav_path,
                segment_seconds=segment_seconds,
                overlap_seconds=resolved_overlap,
                split_mode=resolved_split_mode,
                model_size=resolved_model,
                beam_size=resolved_beam,
                progress_callback=progress_callback,
                vad_max_chunk_seconds=vad_max_chunk_seconds,
                vad_merge_min_seconds=vad_merge_min_seconds,
                vad_target_chunk_seconds=vad_target_chunk_seconds,
            )
            inner = _merge_fw_runtime(
                model, inner, resolved_model, resolved_cpu_threads, resolved_compute_type
            )
            return _finalize(text, inner)

        # 메모리 OOM이면 모델을 small로 재시도 (medium이 "처음 테스트용"이지만,
        # 환경에 따라 medium이 OOM일 수 있어 안전장치를 둡니다).
        oom = "unable to allocate" in msg.lower() or "out of memory" in msg.lower() or "oom" in msg.lower()
        current_model = resolved_model.strip().lower()
        if oom and current_model != "small":
            os.environ["WHISPER_MODEL_SIZE"] = "small"
            try:
                # st.cache_resource wrapper 캐시 제거
                if hasattr(_load_model, "clear"):
                    _load_model.clear()
            except Exception:
                pass

            model = _load_model("small", resolved_cpu_threads, resolved_compute_type)
            text, inner = _transcribe_with_segments(
                model=model,
                wav_path=wav_path,
                segment_seconds=segment_seconds,
                overlap_seconds=resolved_overlap,
                split_mode=resolved_split_mode,
                model_size="small",
                beam_size=resolved_beam,
                progress_callback=progress_callback,
                vad_max_chunk_seconds=vad_max_chunk_seconds,
                vad_merge_min_seconds=vad_merge_min_seconds,
                vad_target_chunk_seconds=vad_target_chunk_seconds,
            )
            inner = _merge_fw_runtime(
                model, inner, "small", resolved_cpu_threads, resolved_compute_type
            )
            return _finalize(text, inner)

        raise RuntimeError(f"STT 모델 로딩/변환 중 오류가 발생했습니다.\n\n상세: {msg}")


def _transcribe_with_segments(
    model: WhisperModel,
    wav_path: str,
    segment_seconds: int,
    overlap_seconds: float = 0.0,
    split_mode: str = "vad",
    *,
    model_size: str,
    beam_size: int,
    progress_callback=None,
    vad_max_chunk_seconds: float | None = None,
    vad_merge_min_seconds: float | None = None,
    vad_target_chunk_seconds: float | None = None,
) -> tuple[str, dict[str, Any]]:
    texts: list[str] = []
    split_mode = (split_mode or "vad").strip().lower()
    use_fixed_split = split_mode == "fixed"
    use_full_once = split_mode == "full"

    overlap_seconds = float(overlap_seconds)
    if overlap_seconds < 0:
        overlap_seconds = 0.0
    seg_f = float(segment_seconds)
    if use_fixed_split and overlap_seconds >= seg_f > 0:
        overlap_seconds = max(0.0, seg_f - 1.0)

    vad_max_resolved: float | None = None
    vad_merge_resolved: float | None = None
    vad_target_resolved: float | None = None
    vad_collect_n = 0
    vad_pack_n = 0

    t_prep_wall0 = time.perf_counter()
    t_load0 = time.perf_counter()
    full_audio = load_mono_16k_wav_as_float32(wav_path)
    prep_load_wav_seconds = time.perf_counter() - t_load0

    base_metrics: dict[str, Any] = {
        "chunk_count": 0,
        "prep_seconds": 0.0,
        "prep_load_wav_seconds": prep_load_wav_seconds,
        "prep_pack_seconds": 0.0,
        "prep_chunk_disk_write_seconds": 0.0,
        "stt_segment_io": "memory",
        "transcribe_seconds": 0.0,
        "chunk_loop_disk_write_seconds": 0.0,
        "chunk_loop_disk_read_seconds": 0.0,
        "chunk_loop_transcribe_seconds": 0.0,
        "chunk_loop_overhead_seconds": 0.0,
        "model_size": model_size,
        "beam_size": beam_size,
        "overlap_seconds": overlap_seconds,
        "split_mode": split_mode,
        "vad_max_chunk_seconds": vad_max_resolved,
        "vad_merge_min_seconds": vad_merge_resolved,
        "vad_target_chunk_seconds": vad_target_resolved,
        "vad_collect_chunk_count": vad_collect_n,
        "vad_pack_chunk_count": vad_pack_n,
    }

    if full_audio.size == 0:
        base_metrics["prep_seconds"] = time.perf_counter() - t_prep_wall0
        return "", base_metrics

    if use_full_once:
        vad_full = os.environ.get("WHISPER_FULL_TRANSCRIBE_VAD_FILTER", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        t_kw0 = time.perf_counter()
        transcribe_kw: dict[str, Any] = {
            "language": "ko",
            "beam_size": beam_size,
            "condition_on_previous_text": False,
            "vad_filter": vad_full,
        }
        if vad_full:
            transcribe_kw["vad_parameters"] = _vad_options_from_env()
        prep_kw_setup = time.perf_counter() - t_kw0
        t_tr0 = time.perf_counter()
        segments, _info = model.transcribe(full_audio, **transcribe_kw)
        chunk_texts = [seg.text for seg in segments if getattr(seg, "text", "").strip()]
        raw_all = "\n".join(chunk_texts).strip()
        t_tr1 = time.perf_counter()
        tr_wall = t_tr1 - t_tr0
        overhead_tail = time.perf_counter() - t_tr1

        base_metrics.update(
            {
                "chunk_count": 1,
                "prep_seconds": prep_load_wav_seconds + prep_kw_setup,
                "prep_pack_seconds": prep_kw_setup,
                "transcribe_seconds": tr_wall,
                "chunk_loop_transcribe_seconds": tr_wall,
                "chunk_loop_overhead_seconds": overhead_tail,
                "vad_collect_chunk_count": 0,
                "vad_pack_chunk_count": 1,
            }
        )
        if progress_callback is not None:
            progress_callback(1.0)
        return raw_all, base_metrics

    t_pack0 = time.perf_counter()
    if use_fixed_split:
        min_chunk = float(os.environ.get("WHISPER_MIN_CHUNK_SECONDS", "0.3"))
        segment_chunks = split_mono_float32_into_segments(
            full_audio,
            float(segment_seconds),
            sample_rate=16000,
            overlap_seconds=overlap_seconds,
            min_chunk_seconds=min_chunk,
        )
    else:
        (
            segment_chunks,
            vad_max_resolved,
            vad_merge_resolved,
            vad_target_resolved,
            vad_collect_n,
            vad_pack_n,
        ) = _vad_pack_audio_to_arrays(
            full_audio,
            max_chunk_seconds=vad_max_chunk_seconds,
            merge_min_seconds=vad_merge_min_seconds,
            target_chunk_seconds=vad_target_chunk_seconds,
        )
        base_metrics["vad_max_chunk_seconds"] = vad_max_resolved
        base_metrics["vad_merge_min_seconds"] = vad_merge_resolved
        base_metrics["vad_target_chunk_seconds"] = vad_target_resolved
        base_metrics["vad_collect_chunk_count"] = vad_collect_n
        base_metrics["vad_pack_chunk_count"] = vad_pack_n

    prep_pack_seconds = time.perf_counter() - t_pack0
    base_metrics["prep_pack_seconds"] = prep_pack_seconds
    base_metrics["prep_seconds"] = time.perf_counter() - t_prep_wall0

    if not segment_chunks:
        return "", base_metrics

    total = len(segment_chunks)
    base_metrics["chunk_count"] = total
    prev_raw_chunk = ""
    loop_write = 0.0
    loop_read = 0.0
    loop_transcribe = 0.0
    loop_overhead = 0.0

    for i, seg_audio in enumerate(segment_chunks):
        t_iter0 = time.perf_counter()
        transcribe_kw = {
            "language": "ko",
            "beam_size": beam_size,
            "condition_on_previous_text": True,
            "vad_filter": use_fixed_split,
        }
        if use_fixed_split:
            transcribe_kw["vad_parameters"] = {
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
            }
        prompt_tail = _tail_for_initial_prompt(prev_raw_chunk)
        if prompt_tail:
            transcribe_kw["initial_prompt"] = prompt_tail
        t_after_kw = time.perf_counter()
        loop_overhead += t_after_kw - t_iter0

        t_tr_start = time.perf_counter()
        segments, _info = model.transcribe(seg_audio, **transcribe_kw)
        chunk_texts = [seg.text for seg in segments if getattr(seg, "text", "").strip()]
        raw_chunk = "\n".join(chunk_texts).strip()
        t_after_segments = time.perf_counter()
        loop_transcribe += t_after_segments - t_tr_start

        if raw_chunk:
            piece = _dedupe_overlap_next_chunk(prev_raw_chunk, raw_chunk) if prev_raw_chunk else raw_chunk
            if piece:
                texts.append(piece)
            prev_raw_chunk = raw_chunk

        t_iter1 = time.perf_counter()
        loop_overhead += t_iter1 - t_after_segments

        if progress_callback is not None and total > 0:
            progress_callback((i + 1) / total)

    base_metrics["chunk_loop_disk_write_seconds"] = loop_write
    base_metrics["chunk_loop_disk_read_seconds"] = loop_read
    base_metrics["chunk_loop_transcribe_seconds"] = loop_transcribe
    base_metrics["chunk_loop_overhead_seconds"] = loop_overhead
    base_metrics["transcribe_seconds"] = loop_transcribe
    return "\n".join(texts).strip(), base_metrics

