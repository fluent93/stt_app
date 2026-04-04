from __future__ import annotations

import os
import tempfile

import streamlit as st
from faster_whisper import WhisperModel

from utils import split_wav_into_segments_with_ffmpeg


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
def _load_model() -> WhisperModel:
    """
    faster-whisper 모델을 1회 로드하고 Streamlit 재실행에서도 재사용합니다.
    - 모델: medium
    - language: ko (transcribe 호출에서 지정)
    """
    model_size = os.environ.get("WHISPER_MODEL_SIZE", "medium").strip() or "medium"

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


def transcribe_korean(wav_path: str, progress_callback=None) -> str:
    """
    wav_path(16kHz, mono WAV 가정)로부터 한국어 STT 수행 후 텍스트 반환.
    - language: ko
    - beam_size: 5
    """
    try:
        segment_seconds = int(os.environ.get("WHISPER_SEGMENT_SECONDS", "10"))
        if segment_seconds < 5:
            segment_seconds = 5

        model = _load_model()

        return _transcribe_with_segments(
            model=model,
            wav_path=wav_path,
            segment_seconds=segment_seconds,
            progress_callback=progress_callback,
        )
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

            segment_seconds = int(os.environ.get("WHISPER_SEGMENT_SECONDS", "10"))
            if segment_seconds < 5:
                segment_seconds = 5

            model = _load_model()
            return _transcribe_with_segments(
                model=model,
                wav_path=wav_path,
                segment_seconds=segment_seconds,
                progress_callback=progress_callback,
            )

        # 메모리 OOM이면 모델을 small로 재시도 (medium이 "처음 테스트용"이지만,
        # 환경에 따라 medium이 OOM일 수 있어 안전장치를 둡니다).
        oom = "unable to allocate" in msg.lower() or "out of memory" in msg.lower() or "oom" in msg.lower()
        current_model = os.environ.get("WHISPER_MODEL_SIZE", "medium").strip().lower() or "medium"
        if oom and current_model != "small":
            segment_seconds = int(os.environ.get("WHISPER_SEGMENT_SECONDS", "10"))
            if segment_seconds < 5:
                segment_seconds = 5

            os.environ["WHISPER_MODEL_SIZE"] = "small"
            try:
                # st.cache_resource wrapper 캐시 제거
                if hasattr(_load_model, "clear"):
                    _load_model.clear()
            except Exception:
                pass

            model = _load_model()
            return _transcribe_with_segments(
                model=model,
                wav_path=wav_path,
                segment_seconds=segment_seconds,
                progress_callback=progress_callback,
            )

        raise RuntimeError(f"STT 모델 로딩/변환 중 오류가 발생했습니다.\n\n상세: {msg}")


def _transcribe_with_segments(
    model: WhisperModel,
    wav_path: str,
    segment_seconds: int,
    progress_callback=None,
) -> str:
    texts: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        seg_paths = split_wav_into_segments_with_ffmpeg(
            wav_path=wav_path,
            segment_seconds=segment_seconds,
            output_dir=tmpdir,
        )

        total = len(seg_paths)
        for i, seg_path in enumerate(seg_paths):
            segments, _info = model.transcribe(
                seg_path,
                language="ko",
                beam_size=5,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 200},
            )

            chunk_texts = [seg.text for seg in segments if getattr(seg, "text", "").strip()]
            chunk_text = "\n".join(chunk_texts).strip()
            if chunk_text:
                texts.append(chunk_text)

            if progress_callback is not None and total > 0:
                # 0~1 범위로 progress_callback 호출
                progress_callback((i + 1) / total)

    return "\n".join(texts).strip()

