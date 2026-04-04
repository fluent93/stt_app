import os
import subprocess
import tempfile
from glob import glob


def _ffmpeg_executable() -> str:
    """환경 변수 FFMPEG_PATH가 있으면 그 경로, 없으면 PATH의 ffmpeg."""
    path = os.environ.get("FFMPEG_PATH", "").strip().strip('"')
    return path if path else "ffmpeg"


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
) -> list[str]:
    """
    WAV를 segment_seconds 단위로 잘라 output_dir에 seg_*.wav 형태로 저장합니다.
    저장된 세그먼트 파일 경로 리스트를 반환합니다.
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"입력 wav 파일을 찾을 수 없습니다: {wav_path}")

    os.makedirs(output_dir, exist_ok=True)

    pattern = os.path.join(output_dir, "seg_%06d.wav")
    ffmpeg_bin = _ffmpeg_executable()

    # WAV PCM으로 재인코딩하며 segment 합니다. (copy는 경계 정밀도/호환성 이슈가 있어 재인코딩이 안정적입니다.)
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


def safe_remove_file(path: str | None) -> None:
    """임시 파일을 안전하게 삭제 (실패해도 무시)."""
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

