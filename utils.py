from __future__ import annotations

import os
import subprocess
import tempfile
import wave
from glob import glob

import numpy as np


def _ffprobe_executable() -> str:
    path = os.environ.get("FFPROBE_PATH", "").strip().strip('"')
    return path if path else "ffprobe"


def _probe_wav_duration_seconds(wav_path: str) -> float:
    """WAV/오디오 파일 길이(초). ffprobe 필요."""
    ffprobe_bin = _ffprobe_executable()
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        wav_path,
    ]
    try:
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "ffprobe 실행 파일을 찾을 수 없습니다. ffmpeg 설치 시 함께 설치되는 경우가 많습니다. "
            "PATH에 추가하거나 FFPROBE_PATH를 지정하세요."
        ) from e
    if completed.returncode != 0:
        err = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"ffprobe로 길이를 읽지 못했습니다. stderr: {err[:1000]}")
    try:
        return float((completed.stdout or "").strip())
    except ValueError as e:
        raise RuntimeError(f"ffprobe duration 파싱 실패: {(completed.stdout or '').strip()!r}") from e


def _ffmpeg_executable() -> str:
    """환경 변수 FFMPEG_PATH가 있으면 그 경로, 없으면 PATH의 ffmpeg."""
    path = os.environ.get("FFMPEG_PATH", "").strip().strip('"')
    return path if path else "ffmpeg"


def load_mono_16k_wav_as_float32(wav_path: str) -> np.ndarray:
    """
    16kHz mono 16-bit PCM WAV를 float32 [-1, 1] numpy 배열로 읽습니다.
    ffmpeg 전처리 직후 파일용으로, PyAV 기반 재디코드 없이 VAD 입력을 만듭니다.
    """
    if not wav_path or not os.path.exists(wav_path):
        raise FileNotFoundError(f"wav 파일을 찾을 수 없습니다: {wav_path}")
    with wave.open(wav_path, "rb") as wf:
        if wf.getnchannels() != 1:
            raise RuntimeError(
                f"VAD 입력 WAV는 모노여야 합니다 (channels={wf.getnchannels()}): {wav_path}"
            )
        if wf.getsampwidth() != 2:
            raise RuntimeError(
                f"VAD 입력 WAV는 16-bit PCM이어야 합니다 (sampwidth={wf.getsampwidth()}): {wav_path}"
            )
        if wf.getframerate() != 16000:
            raise RuntimeError(
                f"VAD 입력 WAV는 16kHz 여야 합니다 (rate={wf.getframerate()}): {wav_path}"
            )
        frames = wf.readframes(wf.getnframes())
    if not frames:
        return np.array([], dtype=np.float32)
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def convert_to_wav_with_ffmpeg(input_path: str) -> str:
    """
    ffmpeg를 사용해 입력을 16kHz, mono WAV로 변환하고
    생성된 WAV 경로를 반환합니다.
    """
    if not input_path or not os.path.exists(input_path):
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_wav_path = tmp_wav.name
    tmp_wav.close()

    ffmpeg_bin = _ffmpeg_executable()
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-f",
        "wav",
        "-y",
        tmp_wav_path,
    ]

    try:
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as e:
        safe_remove_file(tmp_wav_path)
        raise FileNotFoundError(
            "ffmpeg 실행 파일을 찾을 수 없습니다. "
            "ffmpeg를 설치한 뒤 PATH에 추가하거나, "
            "환경 변수 FFMPEG_PATH에 ffmpeg.exe 전체 경로를 지정하세요."
        ) from e

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        safe_remove_file(tmp_wav_path)
        raise RuntimeError(f"ffmpeg 변환 실패입니다. stderr: {stderr[:1000]}")

    return tmp_wav_path


def split_wav_into_segments_with_ffmpeg(
    wav_path: str,
    segment_seconds: int,
    output_dir: str,
    overlap_seconds: float = 0.0,
) -> list[str]:
    """
    WAV를 segment_seconds 단위로 잘라 output_dir에 seg_*.wav 형태로 저장합니다.
    overlap_seconds > 0 이면 hop = segment_seconds - overlap 으로 겹치는 윈도를 만듭니다
    (긴 오디오 경계 품질용). 0이면 기존처럼 ffmpeg segment muxer 한 번으로 처리합니다.
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"입력 wav 파일을 찾을 수 없습니다: {wav_path}")

    os.makedirs(output_dir, exist_ok=True)

    overlap_seconds = float(overlap_seconds)
    if overlap_seconds <= 0.0:
        return _split_wav_disjoint_segments_ffmpeg(
            wav_path=wav_path,
            segment_seconds=segment_seconds,
            output_dir=output_dir,
        )

    if overlap_seconds >= float(segment_seconds):
        raise ValueError(
            f"overlap_seconds({overlap_seconds})는 segment_seconds({segment_seconds})보다 작아야 합니다."
        )

    hop = float(segment_seconds) - overlap_seconds
    duration = _probe_wav_duration_seconds(wav_path)
    ffmpeg_bin = _ffmpeg_executable()
    min_chunk = float(os.environ.get("WHISPER_MIN_CHUNK_SECONDS", "0.3"))

    paths: list[str] = []
    start = 0.0
    idx = 0
    while start < duration:
        chunk_len = min(float(segment_seconds), duration - start)
        if chunk_len < min_chunk:
            break
        out_path = os.path.join(output_dir, f"seg_{idx:06d}.wav")
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            wav_path,
            "-ss",
            str(start),
            "-t",
            str(chunk_len),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            "-y",
            out_path,
        ]
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                "ffmpeg 오버랩 세그먼트 추출 실패입니다. "
                f"stderr: {(completed.stderr or '').strip()[:1000]}"
            )
        paths.append(out_path)
        idx += 1
        if start + chunk_len >= duration - 1e-6:
            break
        start += hop

    if not paths:
        raise RuntimeError("오버랩 세그먼트 분할 결과 파일이 없습니다.")
    return paths


def _split_wav_disjoint_segments_ffmpeg(
    wav_path: str,
    segment_seconds: int,
    output_dir: str,
) -> list[str]:
    """겹침 없이 segment muxer로 분할 (기존 동작)."""
    pattern = os.path.join(output_dir, "seg_%06d.wav")
    ffmpeg_bin = _ffmpeg_executable()
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        wav_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "segment",
        "-segment_time",
        str(int(segment_seconds)),
        "-reset_timestamps",
        "1",
        pattern,
    ]
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "ffmpeg 세그먼트 분할 실패입니다. "
            f"stderr: {(completed.stderr or '').strip()[:1000]}"
        )
    segments = sorted(glob(os.path.join(output_dir, "seg_*.wav")))
    if not segments:
        raise RuntimeError("ffmpeg 세그먼트 분할 결과 파일을 찾지 못했습니다.")
    return segments


def split_mono_float32_into_segments(
    audio: np.ndarray,
    segment_seconds: float,
    *,
    sample_rate: int = 16000,
    overlap_seconds: float = 0.0,
    min_chunk_seconds: float = 0.3,
) -> list[np.ndarray]:
    """
    이미 메모리에 있는 mono float32 [-1,1] 오디오를 고정 길이로 잘라 numpy 배열 목록으로 반환합니다.
    ffmpeg 임시 파일 없이 슬라이스만 사용합니다.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return []
    seg_s = float(segment_seconds)
    if seg_s <= 0:
        return []
    overlap_s = max(0.0, float(overlap_seconds))
    seg_samples = max(1, int(seg_s * sample_rate))
    min_samples = max(1, int(float(min_chunk_seconds) * sample_rate))
    hop_samples = seg_samples
    if overlap_s > 0.0:
        hop_s = seg_s - overlap_s
        if hop_s <= 0:
            raise ValueError("overlap_seconds must be smaller than segment_seconds")
        hop_samples = max(1, int(hop_s * sample_rate))

    out: list[np.ndarray] = []
    start = 0
    n = audio.size
    while start < n:
        end = min(start + seg_samples, n)
        if end - start >= min_samples:
            out.append(np.ascontiguousarray(audio[start:end], dtype=np.float32))
        if end >= n:
            break
        start += hop_samples
    return out


def safe_remove_file(path: str | None) -> None:
    """임시 파일을 안전하게 삭제 (실패해도 무시)."""
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

