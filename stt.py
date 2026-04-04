from __future__ import annotations

import logging
import os
import re
import tempfile
import time
import wave
from typing import Any

import numpy as np
import streamlit as st
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, collect_chunks, get_speech_timestamps

from utils import load_mono_16k_wav_as_float32, split_wav_into_segments_with_ffmpeg

logger = logging.getLogger(__name__)

# Whisper initial_prompt는 토큰 상한(~224)이 있어, 이전 청크 문맥은 꼬리만 넘깁니다.
_CHUNK_CONTEXT_MAX_CHARS = 448

# STT 기본 튜닝(환경변수 미설정 시). 더 빠른 모델은 model_size를 "small" / "base" 등으로 변경.
_STT_DEFAULTS: dict[str, str | int] = {
    "model_size": "medium",
    "beam_size": 2,
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


def _prepare_segment_wav_paths_vad(wav_path: str, output_dir: str) -> list[str]:
    """
    Silero VAD( faster-whisper 내장 )로 음성 구간을 찾고,
    collect_chunks로 max_duration 이하로 묶어 WAV 조각 경로 목록을 만듭니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    audio = load_mono_16k_wav_as_float32(wav_path)
    if audio.size == 0:
        return []

    vad_opt = _vad_options_from_env()
    timestamps = get_speech_timestamps(audio, vad_options=vad_opt)
    if not timestamps:
        return [wav_path]

    max_chunk_s = float(os.environ.get("WHISPER_VAD_MAX_CHUNK_SECONDS", "30"))
    if max_chunk_s < 5.0:
        max_chunk_s = 5.0

    audio_chunks, _meta = collect_chunks(
        audio,
        timestamps,
        sampling_rate=16000,
        max_duration=max_chunk_s,
    )

    min_samples = int(16000 * float(os.environ.get("WHISPER_MIN_CHUNK_SECONDS", "0.3")))
    paths: list[str] = []
    idx = 0
    for chunk_arr in audio_chunks:
        if chunk_arr.size < min_samples:
            continue
        out_path = os.path.join(output_dir, f"seg_{idx:06d}.wav")
        _write_mono_wav_f32(out_path, chunk_arr, 16000)
        paths.append(out_path)
        idx += 1

    return paths if paths else [wav_path]


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
def _load_model(model_size: str) -> WhisperModel:
    """
    faster-whisper 모델을 model_size별로 캐시해 재사용합니다.
    (Streamlit cache_resource — 인자마다 별도 인스턴스)
    """
    model_size = (model_size or "").strip() or str(_STT_DEFAULTS["model_size"])

    # device="auto"로 두되, compute_type을 cpu에도 안전한 값으로 선택합니다.
    # (일부 환경에서 float16이 cpu에서 문제가 될 수 있어 int8 권장)
    download_root = os.environ.get("WHISPER_MODEL_DIR") or None
    local_files_only = os.environ.get("WHISPER_LOCAL_FILES_ONLY", "0").lower() in (
        "1",
        "true",
        "yes",
    )

    # 메모리 OOM을 줄이기 위해 cpu_threads/num_workers를 보수적으로 둡니다.
    return WhisperModel(
        model_size,
        device="cpu",
        cpu_threads=int(os.environ.get("WHISPER_CPU_THREADS", "2")),
        num_workers=1,
        compute_type="int8",
        download_root=download_root,
        local_files_only=local_files_only,
    )


def _env_split_settings() -> tuple[str, int, float]:
    """WHISPER_SPLIT_MODE, 고정 분할용 segment/overlap 초 단위."""
    split_mode = os.environ.get("WHISPER_SPLIT_MODE", "vad").strip().lower()
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
) -> tuple[str, dict[str, Any]]:
    """
    wav_path(16kHz, mono WAV 가정)로부터 한국어 STT 수행.
    model_size / beam_size / overlap_seconds 를 주면 해당 실행에만 적용(환경변수보다 우선).
    반환: (텍스트, 메트릭 dict).
    """
    t_stt_wall = time.perf_counter()
    split_mode, segment_seconds, overlap_env = _env_split_settings()
    resolved_model = (model_size or "").strip() or _env_model_size()
    resolved_beam = max(1, min(int(beam_size) if beam_size is not None else _env_beam_size(), 10))
    resolved_overlap = float(overlap_seconds) if overlap_seconds is not None else overlap_env
    if resolved_overlap < 0:
        resolved_overlap = 0.0

    def _finalize(text: str, inner: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        inner = dict(inner)
        inner["stt_total_seconds"] = time.perf_counter() - t_stt_wall
        n = int(inner.get("chunk_count") or 0)
        tr = float(inner.get("transcribe_seconds") or 0.0)
        inner["avg_chunk_seconds"] = (tr / n) if n else 0.0
        logger.info(
            "STT 완료 wall=%.3fs transcribe=%.3fs prep=%.3fs chunks=%d avg_chunk=%.3fs model=%s beam=%d overlap=%.2f split=%s",
            inner["stt_total_seconds"],
            tr,
            float(inner.get("prep_seconds") or 0.0),
            n,
            inner["avg_chunk_seconds"],
            inner.get("model_size", ""),
            int(inner.get("beam_size") or 0),
            float(inner.get("overlap_seconds") or 0.0),
            inner.get("split_mode", ""),
        )
        return text, inner

    try:
        model = _load_model(resolved_model)

        text, inner = _transcribe_with_segments(
            model=model,
            wav_path=wav_path,
            segment_seconds=segment_seconds,
            overlap_seconds=resolved_overlap,
            split_mode=split_mode,
            model_size=resolved_model,
            beam_size=resolved_beam,
            progress_callback=progress_callback,
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

            model = _load_model(resolved_model)
            text, inner = _transcribe_with_segments(
                model=model,
                wav_path=wav_path,
                segment_seconds=segment_seconds,
                overlap_seconds=resolved_overlap,
                split_mode=split_mode,
                model_size=resolved_model,
                beam_size=resolved_beam,
                progress_callback=progress_callback,
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

            model = _load_model("small")
            text, inner = _transcribe_with_segments(
                model=model,
                wav_path=wav_path,
                segment_seconds=segment_seconds,
                overlap_seconds=resolved_overlap,
                split_mode=split_mode,
                model_size="small",
                beam_size=resolved_beam,
                progress_callback=progress_callback,
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
) -> tuple[str, dict[str, Any]]:
    texts: list[str] = []
    split_mode = (split_mode or "vad").strip().lower()
    use_fixed_split = split_mode == "fixed"

    overlap_seconds = float(overlap_seconds)
    if overlap_seconds < 0:
        overlap_seconds = 0.0
    seg_f = float(segment_seconds)
    if use_fixed_split and overlap_seconds >= seg_f > 0:
        overlap_seconds = max(0.0, seg_f - 1.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        t_prep0 = time.perf_counter()
        if use_fixed_split:
            seg_paths = split_wav_into_segments_with_ffmpeg(
                wav_path=wav_path,
                segment_seconds=segment_seconds,
                output_dir=tmpdir,
                overlap_seconds=overlap_seconds,
            )
        else:
            seg_paths = _prepare_segment_wav_paths_vad(wav_path, tmpdir)
        prep_seconds = time.perf_counter() - t_prep0

        base_metrics: dict[str, Any] = {
            "chunk_count": 0,
            "prep_seconds": prep_seconds,
            "transcribe_seconds": 0.0,
            "model_size": model_size,
            "beam_size": beam_size,
            "overlap_seconds": overlap_seconds,
            "split_mode": split_mode,
        }

        if not seg_paths:
            return "", base_metrics

        total = len(seg_paths)
        base_metrics["chunk_count"] = total
        prev_raw_chunk = ""
        transcribe_seconds = 0.0
        for i, seg_path in enumerate(seg_paths):
            transcribe_kw: dict = {
                "language": "ko",
                "beam_size": beam_size,
                "condition_on_previous_text": True,
                # VAD로 이미 음성 구간만 자른 경우 내부 VAD는 중복이라 끕니다.
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

            t_tr0 = time.perf_counter()
            segments, _info = model.transcribe(seg_path, **transcribe_kw)
            transcribe_seconds += time.perf_counter() - t_tr0

            chunk_texts = [seg.text for seg in segments if getattr(seg, "text", "").strip()]
            raw_chunk = "\n".join(chunk_texts).strip()
            if raw_chunk:
                piece = _dedupe_overlap_next_chunk(prev_raw_chunk, raw_chunk) if prev_raw_chunk else raw_chunk
                if piece:
                    texts.append(piece)
                prev_raw_chunk = raw_chunk

            if progress_callback is not None and total > 0:
                # 0~1 범위로 progress_callback 호출
                progress_callback((i + 1) / total)

        base_metrics["transcribe_seconds"] = transcribe_seconds
        return "\n".join(texts).strip(), base_metrics

