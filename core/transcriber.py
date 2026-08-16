import os
import sys
import json
import tempfile
import shutil
from typing import List, Dict

import whisper
from dotenv import load_dotenv
from sarvamai import SarvamAI


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

ENV_PATH = os.path.join(
    PROJECT_ROOT,
    ".env"
)

load_dotenv(ENV_PATH)


# ============================================================
# AUDIO PROCESSOR
# ============================================================

from utils.audio_processor import (
    download_youtube_audio,
    process_audio_file
)


# ============================================================
# CONFIGURATION
# ============================================================

WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    "base"
)

DEVICE = os.getenv(
    "WHISPER_DEVICE",
    "cpu"
)

SARVAM_API_KEY = os.getenv(
    "SARVAM_API_KEY"
)

SARVAM_MODEL = os.getenv(
    "SARVAM_MODEL",
    "saaras:v3"
)


# ============================================================
# SARVAM LANGUAGE CONFIGURATION
# ============================================================

# Languages that will be routed to Sarvam.
#
# For now, Hindi is the main requirement.
#
# Whisper returns language codes such as:
#
#     en
#     hi
#     bn
#     ta
#     te
#
# We currently route Hindi to Sarvam.
#
# You can add more later if required.

SARVAM_LANGUAGES = {
    "hi"
}


# ============================================================
# WHISPER MODEL CACHE
# ============================================================

_model = None


def load_model():
    """
    Load Whisper model once and reuse it.

    The model is NOT loaded again for every chunk.
    """

    global _model

    if _model is None:

        print()
        print("========================================")
        print("LOADING WHISPER MODEL")
        print("========================================")

        print(f"Model : {WHISPER_MODEL}")
        print(f"Device: {DEVICE}")
        print()

        _model = whisper.load_model(
            WHISPER_MODEL,
            device=DEVICE
        )

        print("Whisper model loaded successfully.")

    return _model


# ============================================================
# SARVAM CLIENT
# ============================================================

_sarvam_client = None


def load_sarvam_client():
    """
    Create Sarvam AI client once and reuse it.
    """

    global _sarvam_client

    if _sarvam_client is None:

        if not SARVAM_API_KEY:
            raise ValueError(
                "SARVAM_API_KEY not found in .env file."
            )

        print()
        print("========================================")
        print("LOADING SARVAM AI")
        print("========================================")

        _sarvam_client = SarvamAI(
            api_subscription_key=SARVAM_API_KEY
        )

        print("Sarvam AI client loaded successfully.")

    return _sarvam_client


# ============================================================
# DETECT LANGUAGE USING WHISPER
# ============================================================

def detect_language(
    audio_path: str
) -> str:
    """
    Detect the language of an audio chunk using Whisper.

    Only language detection is performed here.

    Returns:
        Whisper language code such as:
            en
            hi
            bn
            ta
            etc.
    """

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    model = load_model()

    print()
    print("DETECTING AUDIO LANGUAGE")
    print(f"Audio: {audio_path}")

    # Load audio
    audio = whisper.load_audio(
        audio_path
    )

    # Whisper works with a 30-second window
    audio = whisper.pad_or_trim(
        audio
    )

    # Convert audio to Mel spectrogram
    mel = whisper.log_mel_spectrogram(
        audio
    ).to(model.device)

    # Detect language
    _, probabilities = model.detect_language(
        mel
    )

    language = max(
        probabilities,
        key=probabilities.get
    )

    probability = probabilities[
        language
    ]

    print(
        f"Detected language: {language}"
    )

    print(
        f"Language probability: "
        f"{probability:.2f}"
    )

    return language


# ============================================================
# TRANSCRIBE ENGLISH CHUNK WITH WHISPER
# ============================================================

def transcribe_with_whisper(
    audio_path: str,
    translate: bool = False
) -> Dict:
    """
    Transcribe an audio chunk using Whisper.

    English audio is handled here.

    Args:
        audio_path:
            WAV audio chunk.

        translate:
            False -> Original language.
            True  -> English translation.

    Returns:
        Dictionary containing:
            audio_path
            text
            language
            engine
    """

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    model = load_model()

    print()
    print("----------------------------------------")
    print("WHISPER TRANSCRIPTION")
    print("----------------------------------------")

    print(
        f"Audio: {audio_path}"
    )

    if translate:

        task = "translate"

        print(
            "Mode : Translation → English"
        )

    else:

        task = "transcribe"

        print(
            "Mode : Transcription"
        )

    print()

    result = model.transcribe(
        audio_path,
        task=task,
        fp16=False
    )

    text = result.get(
        "text",
        ""
    ).strip()

    language = result.get(
        "language",
        "unknown"
    )

    print(
        "Whisper transcription completed."
    )

    print(
        f"Detected language: {language}"
    )

    return {
        "audio_path": audio_path,
        "text": text,
        "language": language,
        "engine": "whisper"
    }


# ============================================================
# TRANSCRIBE HINDI CHUNKS USING SARVAM BATCH API
# ============================================================

def transcribe_hindi_chunks_with_sarvam(
    chunk_paths: List[str]
) -> Dict:
    """
    Send Hindi audio chunks to Sarvam Saaras v3
    using the Batch Speech-to-Text API.

    Sarvam translates Hindi speech directly
    into English text.

    Batch API is used because the normal Sarvam
    REST STT endpoint accepts only up to 30 seconds,
    while our audio chunks are 10 minutes long.

    Returns:
        Dictionary:
            chunk_results
            full_text
    """

    if not chunk_paths:
        raise ValueError(
            "No Hindi chunks provided."
        )

    client = load_sarvam_client()

    print()
    print("========================================")
    print("SARVAM AI HINDI → ENGLISH")
    print("========================================")

    print(
        f"Hindi chunks: {len(chunk_paths)}"
    )

    print(
        f"Model: {SARVAM_MODEL}"
    )

    print()

    # --------------------------------------------------------
    # Temporary directory for Sarvam output
    # --------------------------------------------------------

    output_directory = tempfile.mkdtemp(
        prefix="sarvam_output_"
    )

    try:

        # ----------------------------------------------------
        # CREATE BATCH JOB
        # ----------------------------------------------------

        print(
            "Creating Sarvam batch job..."
        )

        job = client.speech_to_text_job.create_job(
            model=SARVAM_MODEL,
            mode="translate"
        )

        print(
            "Sarvam batch job created."
        )

        # ----------------------------------------------------
        # UPLOAD ALL HINDI CHUNKS
        # ----------------------------------------------------

        print()
        print(
            "Uploading Hindi chunks to Sarvam..."
        )

        job.upload_files(
            file_paths=chunk_paths
        )

        print(
            "All Hindi chunks uploaded."
        )

        # ----------------------------------------------------
        # START JOB
        # ----------------------------------------------------

        print()
        print(
            "Starting Sarvam transcription..."
        )

        job.start()

        # ----------------------------------------------------
        # WAIT
        # ----------------------------------------------------

        print()
        print(
            "Waiting for Sarvam processing..."
        )

        job.wait_until_complete(
            poll_interval=5,
            timeout=1800
        )

        print()
        print(
            "Sarvam processing completed."
        )

        # ----------------------------------------------------
        # CHECK RESULTS
        # ----------------------------------------------------

        file_results = job.get_file_results()

        failed_files = file_results.get(
            "failed",
            []
        )

        successful_files = file_results.get(
            "successful",
            []
        )

        if failed_files:

            print()
            print(
                "WARNING: Some Sarvam files failed:"
            )

            for failed in failed_files:

                print(
                    f"  {failed}"
                )

        if not successful_files:

            raise RuntimeError(
                "Sarvam did not successfully "
                "process any Hindi chunks."
            )

        # ----------------------------------------------------
        # DOWNLOAD SARVAM OUTPUT JSON FILES
        # ----------------------------------------------------

        print()
        print(
            "Downloading Sarvam results..."
        )

        job.download_outputs(
            output_dir=output_directory
        )

        # ----------------------------------------------------
        # FIND JSON OUTPUT FILES
        # ----------------------------------------------------

        json_files = []

        for filename in os.listdir(
            output_directory
        ):

            if filename.lower().endswith(
                ".json"
            ):

                json_files.append(
                    filename
                )

        # Sort output files so:
        #
        # 0.json
        # 1.json
        # 2.json
        #
        # remain in original chunk order.

        def json_sort_key(filename):

            name = os.path.splitext(
                filename
            )[0]

            try:
                return int(name)

            except ValueError:
                return name

        json_files.sort(
            key=json_sort_key
        )

        if not json_files:

            raise RuntimeError(
                "Sarvam completed but no JSON "
                "transcription files were found."
            )

        # ----------------------------------------------------
        # READ TRANSCRIPTS
        # ----------------------------------------------------

        chunk_results = []

        for index, json_file in enumerate(
            json_files
        ):

            json_path = os.path.join(
                output_directory,
                json_file
            )

            with open(
                json_path,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(
                    file
                )

            text = data.get(
                "transcript",
                ""
            )

            language = data.get(
                "language_code",
                "hi-IN"
            )

            text = text.strip()

            # ------------------------------------------------
            # Match output back to input chunk
            # ------------------------------------------------

            if index < len(chunk_paths):

                audio_path = chunk_paths[
                    index
                ]

            else:

                audio_path = ""

            chunk_results.append(
                {
                    "chunk_index": index,
                    "audio_path": audio_path,
                    "text": text,
                    "language": language,
                    "engine": "sarvam"
                }
            )

            print()
            print(
                f"Sarvam chunk "
                f"{index + 1}/{len(json_files)} "
                f"completed."
            )

        # ----------------------------------------------------
        # COMBINE SARVAM TEXT
        # ----------------------------------------------------

        full_text = "\n\n".join(
            result["text"]
            for result in chunk_results
            if result["text"]
        )

        print()
        print(
            "Sarvam Hindi → English "
            "translation completed."
        )

        return {
            "chunk_results": chunk_results,
            "full_text": full_text
        }

    finally:

        # ----------------------------------------------------
        # CLEAN TEMPORARY OUTPUT
        # ----------------------------------------------------

        shutil.rmtree(
            output_directory,
            ignore_errors=True
        )


# ============================================================
# TRANSCRIBE ALL CHUNKS
# ============================================================

def transcribe_all(
    chunk_paths: List[str],
    translate: bool = False
) -> Dict:
    """
    Transcribe every audio chunk automatically.

    Pipeline:

        Chunk
          ↓
        Whisper language detection
          ↓
        ┌──────────────┬──────────────┐
        │ English      │ Hindi        │
        ↓              ↓
        Whisper         Sarvam
        ↓              ↓
        English text   English text
        └──────────────┴──────────────┘
                 ↓
           Full transcript

    Hindi chunks are sent to Sarvam Saaras v3
    using the Batch API.

    English chunks are transcribed locally
    using Whisper.

    Args:
        chunk_paths:
            List returned automatically by
            audio_processor.py.

        translate:
            If True:
                Whisper translation is enabled
                for non-Hindi audio.

            Hindi is always translated to English
            through Sarvam.

    Returns:
        Dictionary containing:
            full_text
            chunks
            language
    """

    if not chunk_paths:

        raise ValueError(
            "No audio chunks were provided."
        )

    print()
    print("========================================")
    print("STARTING FULL TRANSCRIPTION")
    print("========================================")

    print(
        f"Total chunks: {len(chunk_paths)}"
    )

    print()

    # --------------------------------------------------------
    # STEP 1 — DETECT LANGUAGE OF EACH CHUNK
    # --------------------------------------------------------

    print()
    print("========================================")
    print("DETECTING CHUNK LANGUAGES")
    print("========================================")

    detected_chunks = []

    for i, audio_path in enumerate(
        chunk_paths,
        start=1
    ):

        print()
        print("----------------------------------------")
        print(
            f"LANGUAGE DETECTION "
            f"{i}/{len(chunk_paths)}"
        )
        print("----------------------------------------")

        language = detect_language(
            audio_path
        )

        detected_chunks.append(
            {
                "chunk_index": i - 1,
                "audio_path": audio_path,
                "language": language
            }
        )

    # --------------------------------------------------------
    # IDENTIFY HINDI / ENGLISH CHUNKS
    # --------------------------------------------------------

    hindi_chunks = [
        chunk
        for chunk in detected_chunks
        if chunk["language"]
        in SARVAM_LANGUAGES
    ]

    non_hindi_chunks = [
        chunk
        for chunk in detected_chunks
        if chunk["language"]
        not in SARVAM_LANGUAGES
    ]

    print()
    print("========================================")
    print("LANGUAGE ROUTING")
    print("========================================")

    print(
        f"Hindi chunks  : {len(hindi_chunks)}"
    )

    print(
        f"Other chunks  : {len(non_hindi_chunks)}"
    )

    # --------------------------------------------------------
    # STEP 2 — WHISPER FOR NON-HINDI CHUNKS
    # --------------------------------------------------------

    results_by_index = {}

    for chunk in non_hindi_chunks:

        index = chunk[
            "chunk_index"
        ]

        audio_path = chunk[
            "audio_path"
        ]

        print()
        print("----------------------------------------")
        print(
            f"PROCESSING CHUNK "
            f"{index + 1}/{len(chunk_paths)}"
        )
        print("----------------------------------------")

        result = transcribe_with_whisper(
            audio_path,
            translate=translate
        )

        result[
            "chunk_index"
        ] = index

        results_by_index[
            index
        ] = result

    # --------------------------------------------------------
    # STEP 3 — SARVAM FOR HINDI CHUNKS
    # --------------------------------------------------------

    if hindi_chunks:

        hindi_paths = [
            chunk["audio_path"]
            for chunk in hindi_chunks
        ]

        sarvam_results = (
            transcribe_hindi_chunks_with_sarvam(
                hindi_paths
            )
        )

        for sarvam_result in sarvam_results[
            "chunk_results"
        ]:

            audio_path = sarvam_result[
                "audio_path"
            ]

            # Find original chunk index
            original_index = None

            for chunk in hindi_chunks:

                if chunk[
                    "audio_path"
                ] == audio_path:

                    original_index = chunk[
                        "chunk_index"
                    ]

                    break

            if original_index is None:

                continue

            sarvam_result[
                "chunk_index"
            ] = original_index

            results_by_index[
                original_index
            ] = sarvam_result

    # --------------------------------------------------------
    # STEP 4 — RESTORE ORIGINAL CHUNK ORDER
    # --------------------------------------------------------

    results = []

    for index in range(
        len(chunk_paths)
    ):

        if index not in results_by_index:

            raise RuntimeError(
                f"No transcription result "
                f"for chunk {index}."
            )

        results.append(
            results_by_index[
                index
            ]
        )

    # --------------------------------------------------------
    # STEP 5 — COMBINE EVERYTHING
    # --------------------------------------------------------

    full_text = "\n\n".join(
        result["text"]
        for result in results
        if result["text"]
    )

    # Determine overall language
    languages = [
        result["language"]
        for result in results
        if result["language"]
    ]

    if languages:

        detected_language = languages[0]

    else:

        detected_language = "unknown"

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("========================================")
    print("TRANSCRIPTION COMPLETE")
    print("========================================")

    print(
        f"Total chunks : {len(results)}"
    )

    print(
        f"Hindi chunks : {len(hindi_chunks)}"
    )

    print(
        f"Other chunks : {len(non_hindi_chunks)}"
    )

    print(
        f"Characters   : {len(full_text)}"
    )

    print()

    print("ENGINE USED PER CHUNK:")

    for result in results:

        print(
            f"Chunk {result['chunk_index'] + 1}: "
            f"{result['engine']} "
            f"({result['language']})"
        )

    return {
        "full_text": full_text,
        "chunks": results,
        "language": detected_language
    }


# ============================================================
# SAVE TRANSCRIPT
# ============================================================

def save_transcript(
    transcript: str,
    output_path: str
) -> str:
    """
    Save the complete transcript to a text file.
    """

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            transcript
        )

    print()
    print("========================================")
    print("TRANSCRIPT SAVED")
    print("========================================")

    print(
        f"File: {output_path}"
    )

    return output_path


# ============================================================
# COMPLETE VIDEO → TRANSCRIPT PIPELINE
# ============================================================

def process_video(
    input_source: str,
    chunk_minutes: int = 10,
    translate: bool = False
) -> Dict:
    """
    Complete pipeline:

        YouTube URL OR local video/audio file
                    ↓
              Audio processor
                    ↓
              WAV conversion
                    ↓
              10-minute chunks
                    ↓
            Language detection
                    ↓
          ┌─────────┴─────────┐
          ↓                   ↓
       English               Hindi
          ↓                   ↓
       Whisper              Sarvam
          ↓                   ↓
          └─────────┬─────────┘
                    ↓
              Full transcript
    """

    print()
    print("========================================")
    print("AI VIDEO TRANSCRIPTION PIPELINE")
    print("========================================")

    # ========================================================
    # STEP 1 — GET AUDIO
    # ========================================================

    if (
        input_source.startswith(
            "http://"
        )
        or input_source.startswith(
            "https://"
        )
    ):

        print()
        print(
            "INPUT TYPE: YOUTUBE / WEB URL"
        )

        downloaded_wav = (
            download_youtube_audio(
                input_source
            )
        )

        input_path = downloaded_wav

    else:

        print()
        print(
            "INPUT TYPE: LOCAL FILE"
        )

        input_path = input_source

    # ========================================================
    # STEP 2 — CONVERT + CHUNK
    # ========================================================

    print()
    print("PROCESSING AUDIO...")

    chunk_paths = process_audio_file(
        input_path,
        chunk_minutes=chunk_minutes
    )

    print()
    print(
        "AUDIO PROCESSING COMPLETE"
    )

    print(
        f"Chunks generated: "
        f"{len(chunk_paths)}"
    )

    # ========================================================
    # STEP 3 — TRANSCRIBE
    # ========================================================

    transcription = transcribe_all(
        chunk_paths,
        translate=translate
    )

    return transcription


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        # ====================================================
        # ONLY INPUT YOU NEED TO PROVIDE
        # ====================================================

        video_url = (
            "https://www.youtube.com/watch?v=TbbSJrY5GqQ&pp=ygUNYmluYXJ5IHNlYXJjaA%3D%3D"
        )

        # ====================================================
        # RUN COMPLETE PIPELINE
        # ====================================================

        result = process_video(
            input_source=video_url,
            chunk_minutes=10,
            translate=False
        )

        # ====================================================
        # DISPLAY FINAL TRANSCRIPT
        # ====================================================

        print()
        print("========================================")
        print("FINAL TRANSCRIPT")
        print("========================================")

        print()
        print(
            result["full_text"]
        )

        # ====================================================
        # SAVE TRANSCRIPT
        # ====================================================

        save_transcript(
            result["full_text"],
            "downloads/transcript.txt"
        )

        print()
        print("========================================")
        print("PIPELINE SUCCESSFUL")
        print("========================================")

    except Exception as e:

        print()
        print("========================================")
        print("ERROR")
        print("========================================")

        print(
            f"{type(e).__name__}: {e}"
        )