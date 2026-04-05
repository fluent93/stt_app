import logging
import os
import tempfile
import time

import streamlit as st

from profile_loader import load_profiles, profile_exec_kw
from stt import default_cpu_threads_for_ui, transcribe_korean
from utils import convert_to_wav_with_ffmpeg, safe_remove_file

_SPLIT_ORDER = ["vad", "fixed", "full"]

_RUN_MODE_NORMAL = "일반 (기존 UI)"
_RUN_MODE_COMPARE = "비교 실험 — small / beam=1 / threads=6 고정"
_RUN_MODE_COMPUTE_SWEEP = "compute_type 실험 — small · full · threads=6 · beam=1"
_COMPUTE_TYPE_SWEEP = ("int8", "float16", "int8_float16")

logger = logging.getLogger(__name__)

# VAD 청크 길이 분포 등 stt 모듈 INFO 로그를 터미널에 출력
_stt_log = logging.getLogger("stt")
if not _stt_log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    _stt_log.addHandler(_h)
    _stt_log.setLevel(logging.INFO)
    _stt_log.propagate = False

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

    try:
        all_profiles = load_profiles()
    except Exception as e:
        st.error(f"profiles.yaml 을 불러오지 못했습니다.\n\n{e}")
        return

    profile_names = list(all_profiles.keys())
    profile_choice = st.selectbox(
        "실행 프로필",
        profile_names,
        help="플랫폼 공통 앱에서 프로필만 바꿔 쓰기 위한 기본 실행 설정입니다.",
    )
    prof = all_profiles[profile_choice]
    profile_kw = profile_exec_kw(prof)
    st.caption(
        f"**{profile_choice}**: model `{prof.get('model_size')}` · compute `{prof.get('compute_type', '-')}` · "
        f"beam `{prof.get('beam_size')}` · threads `{prof.get('cpu_threads')}` · split `{prof.get('split_mode')}` · "
        f"overlap `{prof.get('overlap_seconds', 0)}` · "
        "우선순위: **고급에서 넘긴 명시 값 > 프로필 > 환경변수 > 코드 기본값**"
    )

    is_compare_3way = False
    is_compute_type_sweep = False
    is_ab_experiment = False
    experiment_mode = False
    stt_experiment_kw = {}
    vad_kw: dict = {}
    ui_cpu_threads = int(prof.get("cpu_threads") or default_cpu_threads_for_ui())
    ui_split = str(prof.get("split_mode") or "vad")

    with st.expander("고급 옵션", expanded=False):
        st.caption("비교 실험·VAD 프리셋·분할 모드·실험 모드 등. 여기서 바꾼 값은 **프로필보다 우선**합니다.")

        run_mode = st.radio(
            "실행 모드",
            [_RUN_MODE_NORMAL, _RUN_MODE_COMPARE, _RUN_MODE_COMPUTE_SWEEP],
            horizontal=True,
            help="비교 실험: VAD / full / fixed. FULL일 때 threads 6·8·10·12. "
            "compute_type 실험: 동일 파일로 int8 · float16 · int8_float16 만 바꿔 STT 시간을 표로 비교.",
        )
        is_compare_3way = run_mode == _RUN_MODE_COMPARE
        is_compute_type_sweep = run_mode == _RUN_MODE_COMPUTE_SWEEP

        if is_compare_3way:
            compare_pick = st.radio(
                "3-way 비교 (고정: **model=small · beam=1 · overlap=0** · VAD/fixed는 threads=6)",
                [
                    "1 — VAD (메모리 청크, baseline)",
                    "2 — full (전체 1회 transcribe)",
                    "3 — fixed (고정 길이 슬라이스, 참고)",
                ],
                horizontal=True,
                help="FULL일 때만 아래에서 **cpu_threads** 를 6·8·10·12 로 바꿔 sweet spot을 찾습니다. "
                "같은 파일로 네 번씩 돌려 STT 총 시간·transcribe_seconds 를 비교하세요.",
            )
            if compare_pick.startswith("1"):
                ui_split = "vad"
            elif compare_pick.startswith("2"):
                ui_split = "full"
            else:
                ui_split = "fixed"

            if ui_split == "full":
                ui_cpu_threads = int(
                    st.radio(
                        "FULL 모드 — faster-whisper cpu_threads (sweet spot 실험)",
                        [6, 8, 10, 12],
                        horizontal=True,
                        help="small·beam1·full 고정. STT 총 시간·transcribe_seconds 로 비교. "
                        "스레드 조합마다 모델 캐시가 달라 **첫 실행은 로드 포함**일 수 있어, 동일 threads로 한 번 더 돌리면 순수 추론에 가깝습니다.",
                    )
                )
            else:
                ui_cpu_threads = 6

            st.info(
                f"**이번 실행**: split=`{ui_split}` · **small / beam 1 / threads {ui_cpu_threads} / overlap 0** · "
                "상세 표에서 **prep_load_wav_seconds**, **prep_pack_seconds**, "
                "**chunk_loop_transcribe_seconds**, **chunk_loop_overhead_seconds**, **stt_segment_io** 를 확인하세요."
            )
            experiment_mode = False
            stt_experiment_kw = {}
            vad_kw = {}
            is_ab_experiment = False
        elif is_compute_type_sweep:
            st.info(
                "**고정**: model **small** · split **full** · **cpu_threads 6** · **beam 1** · overlap **0** · "
                f"순서대로 **{' · '.join(_COMPUTE_TYPE_SWEEP)}** 각각 STT 후 표로 비교합니다. "
                "ffmpeg 변환은 1회 공통이며, 표의 시간은 **각 실행의 STT 구간**입니다."
            )
            experiment_mode = False
            stt_experiment_kw = {}
            vad_kw = {}
            is_ab_experiment = False
            ui_cpu_threads = 6
            ui_split = "full"
        else:
            _vad_presets: list[tuple[str, dict]] = [
                ("기본 (환경변수·코드 기본값)", {}),
                (
                    "실험 A — target≈22s, max=30s",
                    {
                        "vad_max_chunk_seconds": 30.0,
                        "vad_target_chunk_seconds": 22.0,
                        "vad_merge_min_seconds": 10.0,
                    },
                ),
                (
                    "실험 B — target≈27s, max=30s",
                    {
                        "vad_max_chunk_seconds": 30.0,
                        "vad_target_chunk_seconds": 27.0,
                        "vad_merge_min_seconds": 10.0,
                    },
                ),
            ]
            _vad_labels = [p[0] for p in _vad_presets]
            vad_choice = st.radio(
                "VAD 청크 패킹 실험",
                _vad_labels,
                horizontal=True,
                help="실험 A/B 선택 시 STT는 **model=small, beam=1, cpu_threads=6, overlap=0, split=vad** 로 고정됩니다. "
                "터미널 로그에 청크 길이(초) 분포가 출력됩니다.",
            )
            vad_kw = dict(_vad_presets[_vad_labels.index(vad_choice)][1])
            is_ab_experiment = vad_choice != _vad_labels[0]

            _cpu_default = int(prof.get("cpu_threads") or default_cpu_threads_for_ui())
            ui_cpu_threads = st.number_input(
                "CPU threads (faster-whisper)",
                min_value=1,
                max_value=32,
                value=_cpu_default,
                step=1,
                key=f"adv_cpu_threads_{profile_choice}",
                disabled=is_ab_experiment,
                help="프로필 기본값으로 초기화됩니다(프로필 전환 시 키가 바뀌어 다시 채워짐). 실험 A/B일 때는 6 고정.",
            )
            if is_ab_experiment:
                st.caption(
                    "실험 A/B 고정: **small** · **beam 1** · **threads 6** · **overlap 0** · **split vad** "
                    "(위 threads 입력은 비활성화됩니다.)"
                )

            experiment_mode = st.checkbox(
                "실험 모드",
                value=False,
                help="모델/beam/overlap을 바꿉니다(**프로필보다 우선**). **실험 A/B 선택 시에는 적용되지 않습니다.**",
                disabled=is_ab_experiment,
            )
            stt_experiment_kw = {}
            if experiment_mode and not is_ab_experiment:
                st.caption("실험 설정은 **이번 변환 실행**에만 반영됩니다.")
                _models = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
                _m0 = str(prof.get("model_size") or "small")
                _mi = _models.index(_m0) if _m0 in _models else 2
                _beam_choices = [1, 2, 3, 5]
                _b0 = int(prof.get("beam_size") or 1)
                _bi = _beam_choices.index(_b0) if _b0 in _beam_choices else 0
                c1, c2, c3 = st.columns(3)
                with c1:
                    ui_model = st.selectbox(
                        "모델 크기",
                        _models,
                        index=_mi,
                        key=f"adv_model_{profile_choice}",
                        help="프로필 기본 모델로 초기화(프로필 전환 시 키 변경).",
                    )
                with c2:
                    ui_beam = st.selectbox(
                        "beam_size",
                        _beam_choices,
                        index=_bi,
                        key=f"adv_beam_{profile_choice}",
                    )
                with c3:
                    ui_overlap = st.slider(
                        "overlap_seconds",
                        min_value=0.0,
                        max_value=5.0,
                        value=float(prof.get("overlap_seconds") or 0.0),
                        step=0.5,
                        key=f"adv_overlap_{profile_choice}",
                        help="WHISPER_SPLIT_MODE=fixed 일 때만 청크 겹침에 사용됩니다.",
                    )
                stt_experiment_kw = {
                    "model_size": ui_model,
                    "beam_size": int(ui_beam),
                    "overlap_seconds": float(ui_overlap),
                }

            if not is_ab_experiment:
                _sd = str(prof.get("split_mode") or "vad")
                _si = _SPLIT_ORDER.index(_sd) if _sd in _SPLIT_ORDER else 0
                ui_split = st.selectbox(
                    "STT 분할 모드",
                    _SPLIT_ORDER,
                    index=_si,
                    key=f"adv_split_{profile_choice}",
                    format_func=lambda k: {
                        "vad": "vad — VAD·메모리 청크 (디스크 세그먼트 없음)",
                        "fixed": "fixed — 고정 길이 numpy 슬라이스",
                        "full": "full — 전체 1회 transcribe (baseline 비교)",
                    }[k],
                    help="프로필 기본 split 으로 초기화. full 은 긴 파일에서 메모리·타임아웃에 유의.",
                )
            else:
                ui_split = "vad"

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

            # 명시 kwargs > 프로필 > 환경변수(stt.py) > 코드 기본값
            if is_compute_type_sweep:
                rows: list[dict[str, object]] = []
                result_text = ""
                timing_last: dict = {}

                def _sweep_progress(i_run: int, ratio: float) -> None:
                    try:
                        ratio = float(ratio)
                    except Exception:
                        ratio = 0.0
                    ratio = max(0.0, min(1.0, ratio))
                    base = (i_run + ratio) / len(_COMPUTE_TYPE_SWEEP)
                    progress_bar.progress(int(70 + base * 25))

                for i_run, ct in enumerate(_COMPUTE_TYPE_SWEEP):
                    progress_text.markdown(
                        f"**3/4: STT ({i_run + 1}/{len(_COMPUTE_TYPE_SWEEP)}) compute_type=`{ct}` …**"
                    )
                    try:
                        txt, timing = transcribe_korean(
                            wav_tmp_path,
                            progress_callback=lambda r, _i=i_run: _sweep_progress(_i, r),
                            model_size="small",
                            beam_size=1,
                            cpu_threads=6,
                            overlap_seconds=0.0,
                            split_mode="full",
                            compute_type=ct,
                        )
                        stt_tot = float(timing.get("stt_total_seconds") or 0.0)
                        tr_sum = float(timing.get("transcribe_seconds") or 0.0)
                        rows.append(
                            {
                                "compute_type": ct,
                                "전체 시간 (s)": round(stt_tot, 3),
                                "STT 시간 (s)": round(tr_sum, 3),
                                "에러": "없음",
                            }
                        )
                        timing_last = dict(timing)
                        if not result_text.strip():
                            result_text = txt
                        logger.info(
                            "compute_sweep ok ct=%s stt_total=%.3fs transcribe=%.3fs",
                            ct,
                            stt_tot,
                            tr_sum,
                        )
                    except Exception as ex:
                        rows.append(
                            {
                                "compute_type": ct,
                                "전체 시간 (s)": "—",
                                "STT 시간 (s)": "—",
                                "에러": str(ex),
                            }
                        )
                        logger.exception("compute_sweep failed ct=%s", ct)

                total_seconds = time.perf_counter() - t_pipeline0

                progress_text.markdown("**4/4: 결과 준비 완료**")
                progress_bar.progress(100)

                st.subheader("compute_type 비교 (동일 파일 · 동일 고정 설정)")
                st.caption(
                    "**전체 시간** = `stt_total_seconds`(로드·prep·인식 포함 STT 구간). "
                    "**STT 시간** = `transcribe_seconds`(Whisper 인식 합계)."
                )
                st.dataframe(rows, hide_index=True, use_container_width=True)

                st.subheader("처리 시간 요약 (이번 버튼 누름 전체)")
                m1, m2 = st.columns(2)
                m1.metric("전체 (업로드~STT 끝)", f"{total_seconds:.2f} s")
                m2.metric("ffmpeg", f"{ffmpeg_seconds:.2f} s")

                if timing_last:
                    st.subheader("마지막 성공 실행의 faster-whisper 설정")
                    fw_dev = str(timing_last.get("inference_device") or "-")
                    fw_ct = str(timing_last.get("inference_compute_type") or "-")
                    fw_th = timing_last.get("inference_cpu_threads", "-")
                    f1, f2, f3 = st.columns(3)
                    f1.metric("device", fw_dev)
                    f2.metric("compute_type", fw_ct)
                    f3.metric("cpu_threads", str(fw_th))

                st.subheader("변환된 텍스트")
                st.text_area(
                    "결과 (첫 성공 실행)",
                    value=result_text or "(성공한 실행이 없거나 결과가 비어 있습니다)",
                    height=300,
                )
                st.download_button(
                    label="결과 텍스트 다운로드 (.txt)",
                    data=result_text or "",
                    file_name="stt_result.txt",
                    mime="text/plain",
                )
            if not is_compute_type_sweep:
                if is_compare_3way:
                    transcribe_kw = {
                        **profile_kw,
                        **{
                            "model_size": "small",
                            "beam_size": 1,
                            "cpu_threads": int(ui_cpu_threads),
                            "overlap_seconds": 0.0,
                            "split_mode": ui_split,
                        },
                    }
                elif is_ab_experiment:
                    transcribe_kw = {
                        **profile_kw,
                        **vad_kw,
                        **{
                            "model_size": "small",
                            "beam_size": 1,
                            "cpu_threads": 6,
                            "overlap_seconds": 0.0,
                            "split_mode": "vad",
                        },
                    }
                else:
                    explicit: dict = {
                        "cpu_threads": int(ui_cpu_threads),
                        "split_mode": ui_split,
                    }
                    if experiment_mode:
                        explicit.update(stt_experiment_kw)
                    transcribe_kw = {**profile_kw, **vad_kw, **explicit}
                result_text, timing = transcribe_korean(
                    wav_tmp_path,
                    progress_callback=_stt_progress,
                    **transcribe_kw,
                )

                total_seconds = time.perf_counter() - t_pipeline0

                logger.info(
                    "파이프라인 완료 total=%.3fs ffmpeg=%.3fs stt_wall=%.3fs transcribe=%.3fs chunks=%d avg_chunk=%.3fs "
                    "model=%s beam=%d overlap=%.2f split=%s | fw_device=%s fw_compute=%s fw_cpu_threads=%s "
                    "fw_model_name=%s ctranslate2_device=%s",
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
                    str(timing.get("inference_device") or ""),
                    str(timing.get("inference_compute_type") or ""),
                    timing.get("inference_cpu_threads", ""),
                    str(timing.get("inference_model_name") or ""),
                    str(timing.get("ctranslate2_device") or "n/a"),
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

                st.subheader("faster-whisper 실행 설정")
                fw_dev = str(timing.get("inference_device") or "-")
                fw_ct = str(timing.get("inference_compute_type") or "-")
                fw_th = timing.get("inference_cpu_threads", "-")
                fw_nm = str(timing.get("inference_model_name") or timing.get("model_size") or "-")
                fw_ct2 = str(timing.get("ctranslate2_device") or "-")
                f1, f2, f3, f4 = st.columns(4)
                f1.metric("device", fw_dev)
                f2.metric("compute_type", fw_ct)
                f3.metric("cpu_threads", str(fw_th))
                f4.metric("model_name", fw_nm)
                st.caption(
                    f"ctranslate2 보고 device: **{fw_ct2}** (M3 등에서 CPU 추론이면 보통 `cpu` 계열로 표시됩니다.)"
                )

                vad_mc = timing.get("vad_max_chunk_seconds")
                vad_mm = timing.get("vad_merge_min_seconds")
                vad_tg = timing.get("vad_target_chunk_seconds")
                vcn = timing.get("vad_collect_chunk_count")
                vpn = timing.get("vad_pack_chunk_count")
                vad_note = ""
                if vad_mc is not None and vad_tg is not None:
                    vad_note = (
                        f" · VAD max **{float(vad_mc):.1f}s** · target **{float(vad_tg):.1f}s**"
                        f" · collect→pack **{int(vcn or 0)}→{int(vpn or 0)}**"
                    )
                    if vad_mm is not None:
                        vad_note += f" · merge_min **{float(vad_mm):.1f}s**"
                st.caption(
                    f"모델(파이프라인) **{timing.get('model_size', '')}** · beam **{int(timing.get('beam_size') or 0)}** · "
                    f"overlap **{float(timing.get('overlap_seconds') or 0):.2f}s** · 분할 `{timing.get('split_mode', '')}`{vad_note} · "
                    f"인식 합계 **{float(timing.get('transcribe_seconds') or 0):.2f}s** · "
                    f"청크당 인식 평균 **{float(timing.get('avg_chunk_seconds') or 0):.3f}s**"
                )

                with st.expander("처리 시간 상세", expanded=False):
                    prep_s = float(timing.get("prep_seconds") or 0.0)
                    tr_s = float(timing.get("transcribe_seconds") or 0.0)
                    pl_wav = float(timing.get("prep_load_wav_seconds") or 0.0)
                    pp_pack = float(timing.get("prep_pack_seconds") or 0.0)
                    cl_w = float(timing.get("chunk_loop_disk_write_seconds") or 0.0)
                    cl_r = float(timing.get("chunk_loop_disk_read_seconds") or 0.0)
                    cl_tr = float(timing.get("chunk_loop_transcribe_seconds") or 0.0)
                    cl_oh = float(timing.get("chunk_loop_overhead_seconds") or 0.0)
                    seg_io = str(timing.get("stt_segment_io") or "—")
                    st.markdown(
                        f"""
| 항목 | 값 |
|------|-----|
| 전체 처리 시간 | **{total_seconds:.3f} s** |
| ffmpeg 변환 시간 | **{ffmpeg_seconds:.3f} s** |
| STT 총 시간 (모델 로드·분할·인식 포함) | **{float(timing.get('stt_total_seconds') or 0):.3f} s** |
| 세그먼트 I/O 방식 | **{seg_io}** (청크 WAV 쓰기/읽기 없음) |
| prep: 전체 WAV 디스크→numpy 로드 | **{pl_wav:.3f} s** |
| prep: VAD·패킹 또는 fixed 슬라이스·full 준비 | **{pp_pack:.3f} s** |
| 분할/VAD 준비 합계 (위 prep_seconds) | {prep_s:.3f} s |
| chunk 루프: 디스크 쓰기 (세그먼트 WAV) | **{cl_w:.3f} s** |
| chunk 루프: 디스크 읽기 (청크별) | **{cl_r:.3f} s** |
| chunk 루프: transcribe+세그먼트 소비 (인퍼런스) | **{cl_tr:.3f} s** |
| chunk 루프: 기타 (프롬프트·중복제거 등) | **{cl_oh:.3f} s** |
| Whisper 인식 시간 (합계, transcribe_seconds) | {tr_s:.3f} s |
| chunk 개수 | **{int(timing.get('chunk_count') or 0)}** |
| chunk당 평균 처리 시간 (인식 합계 ÷ chunk) | **{float(timing.get('avg_chunk_seconds') or 0):.3f} s** |
| 사용 모델 (파이프라인) | **{timing.get('model_size', '')}** |
| beam_size | **{int(timing.get('beam_size') or 0)}** |
| overlap_seconds | **{float(timing.get('overlap_seconds') or 0):.3f}** |
| 분할 모드 | {timing.get('split_mode', '')} |
| VAD max_chunk (초) | {f"**{float(vad_mc):.3f}**" if vad_mc is not None else "— (fixed 분할 등)"} |
| VAD target_chunk (초) | {f"**{float(vad_tg):.3f}**" if vad_tg is not None else "—"} |
| VAD merge_min (초, 잔여 짧은 조각) | {f"**{float(vad_mm):.3f}**" if vad_mm is not None else "—"} |
| VAD 청크 수 (collect_chunks → 패킹 후) | **{int(vcn or 0)}** → **{int(vpn or 0)}** |
| **inference_device** | **{timing.get('inference_device', '-')}** |
| **inference_compute_type** | **{timing.get('inference_compute_type', '-')}** |
| **inference_cpu_threads** | **{timing.get('inference_cpu_threads', '-')}** |
| **inference_model_name** | **{timing.get('inference_model_name', '-')}** |
| ctranslate2_device (로드된 엔진) | {timing.get('ctranslate2_device', '-')} |
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

