# STT App (Streamlit + faster-whisper)

아이폰/녹음 파일을 업로드하면 ffmpeg로 16kHz mono WAV로 변환한 뒤, `faster-whisper`로 한국어 STT를 수행합니다.

- 지원 형식: **m4a, mp3, wav, mp4**
- 출력:
  - **전체 텍스트** (기존)
  - **구조화 전사(세그먼트 타임스탬프 포함)**: `structured_transcript.json`, `structured_transcript_llm.txt`

> 목적: 사람이 바로 읽기 좋은 회의록이 아니라, **LLM(ChatGPT/Claude)이 잘 정리할 수 있는 1차 작업본**을 만드는 데 초점을 둡니다.

---

## 0) Repo

- Repo: `https://github.com/fluent93/stt_app`

---

## 1) 준비물

### 공통
- Git
- Python **3.11**
- ffmpeg (PATH 등록 또는 `FFMPEG_PATH` 설정)

### Windows에서 Python 확인

PowerShell:

```powershell
py -3.11 --version
```

> `py`가 없다면 Python 설치 시 **Add python.exe to PATH** 옵션을 켜고 다시 설치하세요.

---

## 2) 설치 (Windows 권장 절차)

### 2.1 클론

```powershell
git clone https://github.com/fluent93/stt_app
cd stt_app
```

### 2.2 가상환경 (권장)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2.3 패키지 설치

```powershell
py -3.11 -m pip install -r requirements.txt
```

---

## 3) ffmpeg 설정 (중요)

이 앱은 업로드 파일을 WAV로 변환하기 위해 ffmpeg를 사용합니다.

### 방법 A) PATH에 ffmpeg 추가 (영구)

ffmpeg 설치 후 `ffmpeg.exe`가 있는 폴더(예: `C:\ffmpeg\bin`)를 사용자 PATH에 추가한 뒤,
새 PowerShell을 열어 확인합니다.

```powershell
ffmpeg -version
```

### 방법 B) PATH 없이 사용 (회사 PC에서 간단)

PowerShell에서 실행 전에 환경변수로 경로를 지정합니다.

```powershell
$env:FFMPEG_PATH="D:\Downloads\ffmpeg-...\bin\ffmpeg.exe"
```

확인:

```powershell
& $env:FFMPEG_PATH -version
```

ffmpeg 다운로드(Windows): `https://www.gyan.dev/ffmpeg/builds/`

---

## 4) 실행

```powershell
cd D:\stt\stt_app
py -3.11 -m streamlit run app.py
```

실행 후 보통 아래 주소로 열립니다.
- Local URL: `http://localhost:8501`

---

## 5) 결과(다운로드)

앱 실행 후 변환이 끝나면 다음을 다운로드할 수 있습니다.

- `stt_result.txt`
  - 전체 텍스트(기존)
- `structured_transcript.json`
  - 세그먼트 단위 타임스탬프 + 텍스트(추후 diarization을 붙이기 위한 `speaker` 필드 포함, 현재는 `null`)
- `structured_transcript_llm.txt`
  - 한 줄당 `[start - end] text` 형태 (LLM에 붙여넣기 쉬움)

`structured_transcript.json` 예시:

```json
{
  "meta": {
    "format_version": 1,
    "time_reference": "source",
    "sample_rate": 16000,
    "model_size": "small",
    "beam_size": 1,
    "split_mode": "full",
    "overlap_seconds": 0.0
  },
  "segments": [
    {
      "speaker": null,
      "start_sec": 72.4,
      "end_sec": 80.1,
      "start": "00:01:12.400",
      "end": "00:01:20.100",
      "text": "..."
    }
  ]
}
```

---

## 6) Troubleshooting (자주 막히는 부분)

### 6.1 `pip` / `py`가 인식되지 않음
- Windows Store 스텁이 잡히는 경우가 많습니다.
- 아래처럼 항상 `py -3.11 -m pip ...` 형태로 실행하는 것을 권장합니다.

```powershell
py -3.11 -m pip install -r requirements.txt
```

### 6.2 `ffmpeg`를 찾을 수 없음
- PATH에 추가했는지 확인하거나,
- 실행 전에 `FFMPEG_PATH`를 설정하세요.

```powershell
$env:FFMPEG_PATH="...\ffmpeg.exe"
```

### 6.3 모델 다운로드 SSL 오류(회사망)
- 회사 프록시/SSL 정책에 따라 Hugging Face 모델 다운로드가 실패할 수 있습니다.
- 이 경우 **네트워크 정책/프록시 설정**이 필요합니다.

### 6.4 긴 파일이 느리거나 메모리(OOM) 발생
- 먼저 10~15분 파일로 실험 후, 상위 설정만 긴 파일(45~60분)로 검증하는 흐름을 권장합니다.

---

## 7) (추후) diarization 붙이는 위치 (참고)

현재 구조화 전사에는 `speaker`가 **optional(null)** 로 남아 있습니다.
추후 diarization 결과 `(speaker, start_sec, end_sec)`를 얻으면,
각 STT 세그먼트 구간과 **겹침이 가장 큰 화자**를 선택해 `speaker`를 채우는 방식으로 확장 가능합니다.

