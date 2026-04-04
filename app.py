import logging
import os
import tempfile
import time

import streamlit as st

from stt import transcribe_korean
from utils import convert_to_wav_with_ffmpeg, safe_remove_file

logger = logging.getLogger(__name__)

st.set_page_config(page_title="한국어 STT 변환기", layout="centered")


def main() -> None:
    st.markdown("**회의록 삽질 해방의 그날까지...**")
    st.caption("개발자 : 류창한")
    st.divider()
    st.title("한국어 음성 → 텍스트 변환기 (STT)")
    st.caption(
        "m4a, mp3, wav, mp4 업로드 후 16kHz mono WAV로 변환합니다. "
        "긴 파일은 음성 구간(VAD) 기준으로 나눠 한국어 STT를 수행합니다."
    )

    uploaded_file = st.file_uploader(
        "파일 업로드",
        type=["m4a", "mp3", "wav", "mp4"],
        help="지원: m4a, mp3, wav, mp4",
    )

    if uploaded_file is None:
        st.info("파일을 업로드한 후 아래 버튼을 눌러 변환을 시작하세요.")
        return

    if not st.session_state.get("has_run_once", False):
        st.session_state["has_run_once"] = True

    experiment_mode = st.checkbox(
        "실험 모드",
        value=False,
        help="켜면 아래 설정이 이번 실행에만 적용됩니다(환경변수보다 우선).",
    )
    stt_experiment_kw: dict = {}
    if experiment_mode:
        st.caption("실험 설정은 **이번 변환 실행**에만 반영됩니다.")
        c1, c2, c3 = st.columns(3)
        with c1:
            ui_model = st.selectbox(
                "모델 크기",
                ["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                index=3,
                help="faster-whisper 사전 학습 크기",
            )
        with c2:
            ui_beam = st.selectbox("beam_size", [1, 2, 3, 5], index=1)
        with c3:
            ui_overlap = st.slider(
                "overlap_seconds",
                min_value=0.0,
                max_value=5.0,
                value=0.0,
                step=0.5,
                help="WHISPER_SPLIT_MODE=fixed 일 때만 청크 겹침에 사용됩니다.",
            )
        stt_experiment_kw = {
            "model_size": ui_model,
            "beam_size": int(ui_beam),
            "overlap_seconds": float(ui_overlap),
        }

    if st.button("변환 시작", type="primary"):
        progress_text = st.empty()
        progress_bar = st.progress(0)

        original_tmp_path: str | None = None
        wav_tmp_path: str | None = None
        result_text: str = ""

        try:
            t_pipeline0 = time.perf_counter()

            # 1) 업로드 파일 저장
            progress_text.markdown("**1/4: 업로드 파일 임시 저장 중...**")
            progress_bar.progress(10)

            suffix = os.path.splitext(uploaded_file.name)[1].lower() or ".input"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                original_tmp_path = tmp.name

            # 2) ffmpeg로 wav 변환
            progress_text.markdown("**2/4: ffmpeg로 16kHz mono WAV 변환 중...**")
            progress_bar.progress(40)

            t_ffmpeg0 = time.perf_counter()
            wav_tmp_path = convert_to_wav_with_ffmpeg(original_tmp_path)
            ffmpeg_seconds = time.perf_counter() - t_ffmpeg0

            # 3) STT 수행
            progress_text.markdown("**3/4: faster-whisper로 STT 수행 중...**")
            progress_bar.progress(70)

            def _stt_progress(ratio: float) -> None:
                # ratio: 0~1
                try:
                    ratio = float(ratio)
                except Exception:
                    ratio = 0.0
                ratio = max(0.0, min(1.0, ratio))
                # 70~95 구간을 STT 진행에 사용
                progress_bar.progress(int(70 + ratio * 25))

            result_text, timing = transcribe_korean(
                wav_tmp_path,
                progress_callback=_stt_progress,
                **stt_experiment_kw,
            )

            total_seconds = time.perf_counter() - t_pipeline0

            logger.info(
                "파이프라인 완료 total=%.3fs ffmpeg=%.3fs stt_wall=%.3fs transcribe=%.3fs chunks=%d avg_chunk=%.3fs model=%s beam=%d overlap=%.2f split=%s",
                total_seconds,
                ffmpeg_seconds,
                float(timing.get("stt_total_seconds") or 0.0),
                float(timing.get("transcribe_seconds") or 0.0),
                int(timing.get("chunk_count") or 0),
                float(timing.get("avg_chunk_seconds") or 0.0),
                str(timing.get("model_size") or ""),
                int(timing.get("beam_size") or 0),
                float(timing.get("overlap_seconds") or 0.0),
                str(timing.get("split_mode") or ""),
            )

            # 4) 결과 출력/다운로드 준비
            progress_text.markdown("**4/4: 결과 준비 완료**")
            progress_bar.progress(100)

            st.subheader("처리 시간 요약")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("전체", f"{total_seconds:.2f} s")
            m2.metric("ffmpeg", f"{ffmpeg_seconds:.2f} s")
            m3.metric("STT 전체", f"{float(timing.get('stt_total_seconds') or 0):.2f} s")
            m4.metric("chunk 수", str(int(timing.get("chunk_count") or 0)))
            st.caption(
                f"모델 **{timing.get('model_size', '')}** · beam **{int(timing.get('beam_size') or 0)}** · "
                f"overlap **{float(timing.get('overlap_seconds') or 0):.2f}s** · 분할 `{timing.get('split_mode', '')}` · "
                f"인식 합계 **{float(timing.get('transcribe_seconds') or 0):.2f}s** · "
                f"청크당 인식 평균 **{float(timing.get('avg_chunk_seconds') or 0):.3f}s**"
            )

            with st.expander("처리 시간 상세", expanded=False):
                prep_s = float(timing.get("prep_seconds") or 0.0)
                tr_s = float(timing.get("transcribe_seconds") or 0.0)
                st.markdown(
                    f"""
| 항목 | 값 |
|------|-----|
| 전체 처리 시간 | **{total_seconds:.3f} s** |
| ffmpeg 변환 시간 | **{ffmpeg_seconds:.3f} s** |
| STT 총 시간 (모델 로드·분할·인식 포함) | **{float(timing.get('stt_total_seconds') or 0):.3f} s** |
| 분할/VAD 준비 시간 | {prep_s:.3f} s |
| Whisper 인식 시간 (합계) | {tr_s:.3f} s |
| chunk 개수 | **{int(timing.get('chunk_count') or 0)}** |
| chunk당 평균 처리 시간 (인식 합계 ÷ chunk) | **{float(timing.get('avg_chunk_seconds') or 0):.3f} s** |
| 사용 모델 | **{timing.get('model_size', '')}** |
| beam_size | **{int(timing.get('beam_size') or 0)}** |
| overlap_seconds | **{float(timing.get('overlap_seconds') or 0):.3f}** |
| 분할 모드 | {timing.get('split_mode', '')} |
"""
                )

            st.subheader("변환된 텍스트")
            st.text_area("결과", value=result_text or "(결과가 비어 있습니다)", height=300)

            st.download_button(
                label="결과 텍스트 다운로드 (.txt)",
                data=result_text,
                file_name="stt_result.txt",
                mime="text/plain",
            )

        except FileNotFoundError as e:
            st.error(str(e))
        except RuntimeError as e:
            st.error(f"변환/처리 중 오류가 발생했습니다.\n\n상세: {e}")
        except Exception as e:
            st.error(f"알 수 없는 오류가 발생했습니다.\n\n상세: {e}")
        finally:
            # 임시 파일 정리
            safe_remove_file(original_tmp_path)
            safe_remove_file(wav_tmp_path)
            progress_text.empty()


if __name__ == "__main__":
    main()

