import os
import tempfile

import streamlit as st

from stt import transcribe_korean
from utils import convert_to_wav_with_ffmpeg, safe_remove_file


st.set_page_config(page_title="한국어 STT 변환기", layout="centered")


def main() -> None:
    st.markdown("**회의록 삽질 해방의 그날까지...**")
    st.caption("개발자 : 류창한")
    st.divider()
    st.title("한국어 음성 → 텍스트 변환기 (STT)")
    st.caption("m4a, mp3, wav, mp4 업로드 후 16kHz mono WAV로 변환하고 한국어 STT를 수행합니다.")

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

    if st.button("변환 시작", type="primary"):
        progress_text = st.empty()
        progress_bar = st.progress(0)

        original_tmp_path: str | None = None
        wav_tmp_path: str | None = None
        result_text: str = ""

        try:
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

            wav_tmp_path = convert_to_wav_with_ffmpeg(original_tmp_path)

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

            result_text = transcribe_korean(wav_tmp_path, progress_callback=_stt_progress)

            # 4) 결과 출력/다운로드 준비
            progress_text.markdown("**4/4: 결과 준비 완료**")
            progress_bar.progress(100)

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

