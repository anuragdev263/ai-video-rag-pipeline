# 🎥 AI Video RAG Pipeline

> An AI-powered video understanding pipeline that transforms YouTube video content into searchable knowledge through audio extraction, multilingual transcription, translation, and Retrieval-Augmented Generation.

**Status:** 🚧 Active Development

---

## 📌 Overview

**AI Video RAG Pipeline** is an end-to-end AI engineering project designed to transform long-form video content into structured, searchable knowledge.

The current system accepts a YouTube video URL, extracts its audio, converts the audio into WAV format, splits long recordings into manageable chunks, automatically detects the spoken language, and routes Hindi audio through **Sarvam AI** for Hindi-to-English transcription.

The next stage of development will extend this pipeline into a complete **Retrieval-Augmented Generation (RAG)** system capable of retrieving relevant information from video transcripts and generating context-aware answers.

---

## 🚀 Current Pipeline

```text
                    YouTube Video URL
                           │
                           ▼
                  ┌──────────────────┐
                  │   yt-dlp         │
                  │ Audio Extraction │
                  └────────┬─────────┘
                           │
                           ▼
                    M4A Audio File
                           │
                           ▼
                  ┌──────────────────┐
                  │     pydub        │
                  │  M4A → WAV       │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  Audio Chunking  │
                  │  10-min chunks   │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  OpenAI Whisper  │
                  │ Language Detect  │
                  └────────┬─────────┘
                           │
                  ┌────────┴─────────┐
                  │                  │
                Hindi              Other
                  │                  │
                  ▼                  ▼
           ┌──────────────┐    Transcription
           │  Sarvam AI   │       Route
           │ Hindi → Eng  │
           └──────┬───────┘
                  │
                  └────────┬─────────┘
                           ▼
                  Final Transcript
                           │
                           ▼
                 downloads/transcript.txt
```

---

# ✨ Features

### 🎬 YouTube Audio Extraction

Downloads the audio stream from a YouTube video using `yt-dlp`.

### 🎵 Audio Processing

Converts downloaded M4A audio into WAV format for downstream speech-processing tasks.

### ✂️ Long Audio Chunking

Long recordings are divided into smaller audio segments.

Current configuration:

* **Chunk duration:** 10 minutes
* **Example video:** ~44 minutes
* **Chunks generated:** 5

### 🌐 Automatic Language Detection

Each audio chunk is analyzed using **Whisper** to determine the spoken language before transcription.

### 🇮🇳 Hindi → English Processing

Hindi audio is automatically routed through **Sarvam AI**.

Current Sarvam model:

```text
saaras:v3
```

### 📝 Transcript Generation

The processed chunks are converted into a final English transcript and saved locally.

```text
downloads/transcript.txt
```

### 🔐 Environment-Based Configuration

API credentials are loaded through environment variables rather than hard-coded into the source code.

---

# 🧪 Current Successful Test

The pipeline has been successfully tested on a YouTube DSA lecture covering the **Binary Search Algorithm**.

### Processing Results

```text
Input Type       : YouTube / Web URL
Audio Chunks     : 5
Chunk Duration   : 10 minutes
Detected Language: Hindi
Hindi Chunks     : 5
Other Chunks     : 0
Sarvam Model     : saaras:v3
Transcript       : Generated successfully
Characters       : 44,265
```

The final pipeline completed successfully and saved the generated transcript to:

```text
downloads/transcript.txt
```

---

# 🛠️ Technology Stack

| Layer                  | Technology                 |
| ---------------------- | -------------------------- |
| Language               | Python                     |
| Video/Audio Extraction | yt-dlp                     |
| Audio Processing       | pydub                      |
| Speech Processing      | OpenAI Whisper             |
| Language Detection     | Whisper                    |
| Hindi → English        | Sarvam AI                  |
| Sarvam Model           | saaras:v3                  |
| Audio Formats          | M4A / WAV                  |
| Environment            | Python Virtual Environment |
| Version Control        | Git / GitHub               |

---

# 📂 Project Structure

```text
ai-video-rag-pipeline/
│
├── core/
│   └── transcriber.py
│
├── utils/
│   └── audio_processor.py
│
├── .gitignore
├── Requirements.txt
└── README.md
```

### `core/transcriber.py`

Responsible for the complete transcription workflow:

* Input handling
* Audio processing orchestration
* Language detection
* Language-based routing
* Sarvam AI processing
* Transcript generation

### `utils/audio_processor.py`

Responsible for:

* YouTube audio downloading
* Audio conversion
* Audio chunk creation

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/anuragdev263/ai-video-rag-pipeline.git
```

```bash
cd ai-video-rag-pipeline
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```bash
pip install -r Requirements.txt
```

---

# 🔑 Environment Configuration

Create a local `.env` file in the project root.

Example:

```env
SARVAM_API_KEY=your_api_key_here
```

> ⚠️ Never commit your actual `.env` file or API credentials to GitHub.

The repository's `.gitignore` is configured to prevent environment secrets, virtual environments, generated audio, and temporary files from being committed.

---

# ▶️ Running the Pipeline

## Audio Processing

Run:

```bash
python utils/audio_processor.py
```

This performs:

```text
YouTube URL
     ↓
Audio Download
     ↓
M4A → WAV
     ↓
Audio Chunking
```

---

## Full Transcription Pipeline

Run:

```bash
python core/transcriber.py
```

The pipeline performs:

```text
Audio Processing
       ↓
Language Detection
       ↓
Language Routing
       ↓
Transcription / Translation
       ↓
Transcript Generation
```

The generated transcript is stored locally as:

```text
downloads/transcript.txt
```

---

# 🧠 RAG Architecture — Next Stage

The current transcription system forms the ingestion layer for the planned RAG architecture.

The target architecture is:

```text
                  YouTube Video
                       │
                       ▼
                Audio Extraction
                       │
                       ▼
                 Transcription
                       │
                       ▼
              Transcript Chunks
                       │
                       ▼
                  Embeddings
                       │
                       ▼
                Vector Database
                       │
                       │
                 User Question
                       │
                       ▼
                Semantic Search
                       │
                       ▼
              Relevant Context
                       │
                       ▼
                     LLM
                       │
                       ▼
              Grounded Answer
```

---

# 🗺️ Development Roadmap

## Phase 1 — Video Ingestion

**Status: ✅ Completed**

* [x] YouTube URL input
* [x] Audio extraction
* [x] M4A → WAV conversion
* [x] Long-audio chunking

---

## Phase 2 — Multilingual Transcription

**Status: ✅ Completed**

* [x] Automatic language detection
* [x] Hindi language routing
* [x] Sarvam AI integration
* [x] Hindi → English processing
* [x] Transcript generation

---

## Phase 3 — Knowledge Processing

**Status: 🔄 Next**

* [ ] Intelligent transcript chunking
* [ ] Metadata generation
* [ ] Embedding generation
* [ ] Vector database integration
* [ ] Semantic similarity search

---

## Phase 4 — Retrieval-Augmented Generation

**Status: 📋 Planned**

* [ ] User question processing
* [ ] Relevant chunk retrieval
* [ ] Context construction
* [ ] LLM integration
* [ ] Grounded answer generation
* [ ] Source/context references

---

## Phase 5 — AI Video Assistant

**Status: 📋 Planned**

* [ ] Interactive chat interface
* [ ] Video-aware question answering
* [ ] Timestamp-aware retrieval
* [ ] Multi-video knowledge bases
* [ ] Web application
* [ ] API layer
* [ ] Production deployment

---

# 🎯 Project Goals

The long-term objective is to build an **AI Video Knowledge Assistant** capable of understanding long-form video content and answering questions using information grounded in the actual video.

The project explores practical implementation of:

* Artificial Intelligence
* Speech Recognition
* Multilingual AI
* Natural Language Processing
* Semantic Search
* Vector Databases
* Retrieval-Augmented Generation
* Large Language Models
* AI Agents
* Production-oriented Python development

---

# 🔒 Security & Git Hygiene

The following files and generated assets are intentionally excluded from version control:

```text
.env
.venv/
__pycache__/
*.pyc
*.wav
*.m4a
*.mp3
*.mp4
*.webm
downloads/
```

This keeps the public repository lightweight and prevents accidental exposure of credentials or large generated media files.

---

# 📊 Current Architecture

```text
┌─────────────────────────────────────────────┐
│              VIDEO INGESTION                │
│                                             │
│              YouTube URL                    │
│                   │                         │
│                   ▼                         │
│               yt-dlp                        │
│                   │                         │
│                   ▼                         │
│              Audio File                     │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│              AUDIO PROCESSING               │
│                                             │
│             M4A → WAV                       │
│                   │                         │
│                   ▼                         │
│            10-minute chunks                 │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│          LANGUAGE INTELLIGENCE              │
│                                             │
│              Whisper                        │
│                │                            │
│       ┌────────┴────────┐                   │
│       ▼                 ▼                   │
│     Hindi             Other                 │
│       │                 │                   │
│       ▼                 ▼                   │
│   Sarvam AI        Transcription            │
│       │                                     │
│       ▼                                     │
│ English Transcript                          │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│             KNOWLEDGE LAYER                 │
│                                             │
│        Transcript → Chunks                  │
│                  ↓                          │
│             Embeddings                       │
│                  ↓                          │
│          Vector Database                     │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│                 RAG                         │
│                                             │
│             User Question                    │
│                  ↓                          │
│          Semantic Retrieval                  │
│                  ↓                          │
│         Relevant Video Context               │
│                  ↓                          │
│                 LLM                          │
│                  ↓                          │
│          Grounded Answer                    │
└─────────────────────────────────────────────┘
```

---

# 📈 Project Status

| Component             | Status     |
| --------------------- | ---------- |
| YouTube ingestion     | ✅ Complete |
| Audio extraction      | ✅ Complete |
| Audio conversion      | ✅ Complete |
| Audio chunking        | ✅ Complete |
| Language detection    | ✅ Complete |
| Hindi routing         | ✅ Complete |
| Sarvam integration    | ✅ Complete |
| Transcript generation | ✅ Complete |
| Transcript chunking   | 🔄 Next    |
| Embeddings            | 📋 Planned |
| Vector database       | 📋 Planned |
| Semantic retrieval    | 📋 Planned |
| RAG                   | 📋 Planned |
| AI chatbot            | 📋 Planned |

---

# 👨‍💻 Author

**Anurag Anand**

Computer Science & Engineering

GitHub: [@anuragdev263](https://github.com/anuragdev263)

---

# ⭐ Contributing

This project is currently under active development.

Suggestions, improvements, and contributions are welcome as the project evolves toward a complete AI-powered video knowledge system.

---

