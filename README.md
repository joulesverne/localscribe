# Interactive Audio Transcription with Speaker Identification

A local audio transcription solution with interactive speaker identification, no API usage or authentication required.

## Features

- **Local Processing:** Run everything on your local machine without cloud services
- **Interactive Speaker Identification:** Plays audio samples from each speaker and lets you provide their names
- **Multi-format Support:** Works with common audio formats (WAV, MP3, FLAC, M4A, etc.)
- **Format Conversion:** Converts any audio format to WAV when needed

## Requirements

- **Python 3.7+**
- **FFmpeg:** Required for audio format handling beyond WAV
  - **Ubuntu/Debian:** `sudo apt-get install ffmpeg`
  - **macOS:** `brew install ffmpeg`
  - **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html)
- **GPU (Optional):** While the script runs on CPU, a GPU will significantly speed up processing

## Installation


1. **Install Python Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Transcribe Audio with Interactive Speaker Identification

```bash
python transcribe.py path/to/your_audio_file.mp3 --output transcript.txt
```

The script will:
1. Process the audio file and identify different speakers
2. Play audio samples of each speaker for you to identify
3. Prompt you to enter names for each speaker
4. Generate a complete transcript with proper speaker names

If a sample is unclear, you can press Enter to hear another sample from the same speaker.

### Command Line Arguments

- **`audio_file`**: Path to your input audio file (supports WAV, MP3, FLAC, M4A, etc.)
- **`--output` or `-o`** (Optional): Path to the output transcript file (default: `transcript.txt`)
- **`--whisper_model`** (Optional): Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) (default: `small`)
- **`--min_speakers`** (Optional): Minimum number of speakers to detect
- **`--max_speakers`** (Optional): Maximum number of speakers to detect
- **`--convert_audio`** (Optional): Explicitly convert audio to WAV format before processing (default: auto-detect)
- **`--samples_per_speaker`** (Optional): Number of audio samples to extract per speaker (default: 3)

## Troubleshooting

- **FFmpeg Not Found:**  
  Ensure FFmpeg is installed and available in your system's PATH. This is required for the audio format conversion.

- **No Sound During Sample Playback:**  
  Make sure your system's audio is working and not muted. The pygame library is used to play audio samples.

- **Speaker Identification Issues:**  
  If the system isn't correctly identifying different speakers, try:  
  - Specifying the exact number of speakers with `--min_speakers` and `--max_speakers`  
  - Using higher quality audio with less background noise  
  - For longer recordings, consider processing smaller sections separately

- **Performance:**  
  If transcription is too slow, consider using a smaller Whisper model (e.g., `small` or `base`) by specifying the `--whisper_model` argument.

- **M4A File Compatibility:**  
  If you encounter issues with M4A files, the `--convert_audio` flag will explicitly convert them to WAV format before processing.

## License

This project is open-source and available under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **OpenAI Whisper:** For providing advanced transcription capabilities
- **SpeechBrain:** For speaker recognition and diarization models
- **FFmpeg:** For robust audio format conversion
