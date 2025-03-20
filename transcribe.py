#!/usr/bin/env python3
"""
Interactive Audio Transcription with Speaker Identification

This script takes an input audio file and produces a transcript with named speakers.
It first identifies different speakers through diarization, then plays a sample
of each speaker's voice and prompts the user to provide their name.

Requirements:
- Python 3.7+
- Install required packages:
    pip install git+https://github.com/openai/whisper.git
    pip install speechbrain soundfile librosa numpy torch pygame

Usage:
    python transcribe_speakerid.py path/to/audio_file --output transcript.txt
"""

import argparse
import numpy as np
import whisper
import soundfile as sf
import librosa
import torch
import os
import sys
import tempfile
import pygame
import subprocess
from speechbrain.inference.speaker import SpeakerRecognition
import torchaudio
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from tqdm import tqdm

def transcribe_segment(audio_data, sample_rate, start, end, model):
    """
    Transcribe a segment of the audio using Whisper.
    
    Parameters:
      audio_data (np.ndarray): The full audio signal.
      sample_rate (int): Original sample rate of the audio.
      start (float): Start time in seconds.
      end (float): End time in seconds.
      model: Loaded Whisper model.
      
    Returns:
      str: Transcribed text.
    """
    # Convert times to sample indices
    start_idx = int(start * sample_rate)
    end_idx = int(end * sample_rate)
    segment_audio = audio_data[start_idx:end_idx]

    # Convert multi-channel audio to mono if necessary
    if segment_audio.ndim > 1:
        segment_audio = np.mean(segment_audio, axis=1)

    # Resample segment if sample rate is not 16kHz
    target_sr = 16000
    if sample_rate != target_sr:
        segment_audio = librosa.resample(segment_audio, orig_sr=sample_rate, target_sr=target_sr)
        sr_for_whisper = target_sr
    else:
        sr_for_whisper = sample_rate

    # Ensure the audio is in float32
    segment_audio = segment_audio.astype(np.float32)

    # Transcribe with Whisper (forcing English for consistency)
    result = model.transcribe(segment_audio, fp16=False, language='en')
    return result.get("text", "").strip()


def convert_to_wav(input_file):
    """Convert any audio file to WAV format for better compatibility.
    
    Parameters:
        input_file (str): Path to the input audio file
        
    Returns:
        str: Path to the converted WAV file
    """
    print(f"Converting {input_file} to WAV format for better compatibility...")
    try:
        # Create a temporary file for the WAV output
        temp_dir = tempfile.mkdtemp()
        output_file = os.path.join(temp_dir, "converted_audio.wav")
        
        # Use ffmpeg for the conversion
        subprocess.run([
            "ffmpeg", 
            "-i", input_file, 
            "-ar", "16000",  # Resample to 16kHz
            "-ac", "1",      # Convert to mono
            "-y",            # Overwrite output file if it exists
            output_file
        ], check=True, capture_output=True)
        
        print(f"Conversion successful: {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"Error converting audio file: {e}")
        print("FFmpeg stderr:", e.stderr.decode())
        print("Continuing with original file...")
        return input_file
    except Exception as e:
        print(f"Unexpected error during conversion: {e}")
        print("Continuing with original file...")
        return input_file


def perform_diarization(audio_file, min_speakers=None, max_speakers=None):
    """Perform speaker diarization on an audio file.
    
    Parameters:
        audio_file (str): Path to audio file
        min_speakers (int, optional): Minimum number of speakers
        max_speakers (int, optional): Maximum number of speakers
        
    Returns:
        list: List of (start_time, end_time, speaker_id) tuples
    """
    print("Loading SpeechBrain speaker recognition model...")
    try:
        # Convert audio to WAV format first to ensure compatibility
        wav_file = convert_to_wav(audio_file)
        
        # Load the SpeechBrain speaker recognition model
        speaker_model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
        
        # Load the audio file
        signal, sample_rate = torchaudio.load(wav_file)
        
        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            signal = resampler(signal)
            sample_rate = 16000
        
        # Calculate total duration
        duration = signal.shape[1] / sample_rate
        print(f"Audio duration: {duration:.2f} seconds")
        
        # Define segment length in seconds
        segment_len = 2.0  # 2-second segments
        hop_len = 1.0  # 1-second overlap
        
        # Extract embeddings from audio segments
        embeddings = []
        timestamps = []
        
        print("Extracting speaker embeddings...")
        for start_sec in tqdm(np.arange(0, duration - segment_len, hop_len)):
            end_sec = start_sec + segment_len
            
            # Convert seconds to samples
            start_sample = int(start_sec * sample_rate)
            end_sample = int(end_sec * sample_rate)
            
            # Extract segment
            segment = signal[:, start_sample:end_sample]
            
            # Get embedding
            with torch.no_grad():
                embed = speaker_model.encode_batch(segment)
                
            # Store embedding and timestamp
            embeddings.append(embed.squeeze().cpu().numpy())
            timestamps.append((start_sec, end_sec))
        
        # Skip diarization if no embeddings were extracted
        if not embeddings:
            print("No embeddings extracted, returning single speaker segment")
            return [(0.0, duration, "spk_0")]
        
        # Convert to numpy array
        embeddings = np.array(embeddings)
        
        # Determine number of speakers
        num_speakers = 2  # Default
        
        if min_speakers is not None and max_speakers is not None:
            if min_speakers == max_speakers:
                num_speakers = min_speakers
            else:
                # Use agglomerative clustering to estimate number of speakers
                # This is a simplified approach
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=0.7,  # Adjust as needed
                    affinity='cosine',
                    linkage='average'
                ).fit(embeddings)
                num_speakers = len(set(clustering.labels_))
                num_speakers = max(min(num_speakers, max_speakers), min_speakers)
        elif min_speakers is not None:
            num_speakers = max(2, min_speakers)
        elif max_speakers is not None:
            num_speakers = min(max_speakers, 8)  # Default upper limit
            
        print(f"Clustering into {num_speakers} speakers...")
        
        # Perform clustering
        clustering = AgglomerativeClustering(
            n_clusters=num_speakers
        )
        labels = clustering.fit_predict(embeddings)
        
        # Post-process segments (merge consecutive segments with same speaker)
        segments = []
        current_speaker = None
        current_start = None
        
        for i, (start_sec, end_sec) in enumerate(timestamps):
            speaker = f"spk_{labels[i]}"
            
            if i == 0:  # First segment
                current_speaker = speaker
                current_start = start_sec
            elif speaker != current_speaker:  # New speaker
                segments.append((current_start, start_sec, current_speaker))
                current_speaker = speaker
                current_start = start_sec
                
        # Add final segment
        if current_start is not None and current_speaker is not None:
            segments.append((current_start, timestamps[-1][1], current_speaker))
            
        print(f"Diarization complete, found {len(segments)} segments")
        return segments
        
    except Exception as e:
        print(f"\nError performing diarization: {str(e)}")
        print("\nFalling back to single-speaker transcription.")
        
        # Get audio duration and return as a single speaker segment
        try:
            duration = librosa.get_duration(path=audio_file)
        except:
            # If that fails, try a more robust approach
            info = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_file], 
                               capture_output=True, text=True)
            duration = float(info.stdout.strip())
            
        return [(0.0, duration, "spk_0")]


def extract_speaker_samples(audio_file, speaker_segments, max_sample_length=5, samples_per_speaker=3):
    """
    Extract multiple representative audio samples for each unique speaker.
    
    Parameters:
        audio_file (str): Path to the audio file
        speaker_segments (list): List of (start, end, speaker_id) tuples
        max_sample_length (int): Maximum length of each sample in seconds
        samples_per_speaker (int): Number of samples to extract per speaker
        
    Returns:
        dict: Mapping of speaker_id to list of sample file paths
    """
    print("Extracting representative samples for each speaker...")
    
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Group segments by speaker
    speaker_to_segments = {}
    for start, end, speaker_id in speaker_segments:
        if speaker_id not in speaker_to_segments:
            speaker_to_segments[speaker_id] = []
        speaker_to_segments[speaker_id].append((start, end))
    
    # Create a temporary directory to store samples
    temp_dir = tempfile.mkdtemp()
    
    # Extract multiple segments for each speaker
    speaker_samples = {}
    for speaker_id, segments in speaker_to_segments.items():
        # Sort segments by duration (longest first)
        segments.sort(key=lambda x: x[1] - x[0], reverse=True)
        
        # Get up to samples_per_speaker segments
        speaker_samples[speaker_id] = []
        for i, (start, end) in enumerate(segments[:samples_per_speaker]):
            # Limit to max_sample_length
            if end - start > max_sample_length:
                end = start + max_sample_length
            
            # Extract the audio segment
            start_idx = int(start * sr)
            end_idx = int(end * sr)
            segment_audio = y[start_idx:end_idx]
            
            # Save to a temporary WAV file
            sample_path = os.path.join(temp_dir, f"{speaker_id}_sample{i+1}.wav")
            sf.write(sample_path, segment_audio, sr)
            
            speaker_samples[speaker_id].append(sample_path)
    
    return speaker_samples


def play_audio_sample(audio_file):
    """
    Play an audio sample using pygame.
    
    Parameters:
        audio_file (str): Path to the audio file to play
    """
    # Initialize pygame mixer
    pygame.mixer.init()
    
    # Load and play the audio file
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    
    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    
    # Clean up
    pygame.mixer.quit()


def identify_speakers(speaker_samples):
    """
    Play sample audio for each speaker and prompt user for names.
    If user doesn't provide a name, try another sample if available.
    
    Parameters:
        speaker_samples (dict): Mapping of speaker_id to list of sample file paths
        
    Returns:
        dict: Mapping of speaker_id to speaker name
    """
    print("\n" + "="*60)
    print("SPEAKER IDENTIFICATION".center(60))
    print("="*60)
    print("I'll play a sample of each speaker's voice.")
    print("Please provide a name for each speaker.")
    print("If the sample is unclear, just press Enter and I'll play another sample.")
    print("-"*60)
    
    speaker_names = {}
    
    for i, (speaker_id, sample_paths) in enumerate(speaker_samples.items(), 1):
        default_name = f"Speaker {i}"
        
        # Try each sample until user provides a name or we run out of samples
        for j, sample_path in enumerate(sample_paths, 1):
            print(f"\nSpeaker {i} sample {j}/{len(sample_paths)}:")
            
            # Play the sample
            play_audio_sample(sample_path)
            
            # Ask for speaker name
            name = input(f"Enter name for {default_name} (or press Enter for another sample): ").strip()
            
            # If name provided, use it and move to next speaker
            if name:
                speaker_names[speaker_id] = name
                break
                
            # If this was the last sample, use default name
            if j == len(sample_paths):
                print(f"No more samples available. Using default name '{default_name}'.")
                speaker_names[speaker_id] = default_name
                break
            
            # Otherwise, continue to next sample
            print("Trying another sample...")
        
    print("\nSpeaker identification complete!")
    print("-"*60)
    
    return speaker_names


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Audio Transcription with Speaker Identification"
    )
    parser.add_argument("audio_file", help="Path to input audio file (WAV, MP3, FLAC, M4A, etc.)")
    parser.add_argument(
        "--output", "-o", help="Path to output transcript file", default="transcript.txt"
    )
    parser.add_argument(
        "--whisper_model",
        help="Whisper model size (e.g., small, medium, large).",
        default="small",
    )
    parser.add_argument(
        "--min_speakers",
        type=int,
        help="Minimum number of speakers (optional)",
        default=None
    )
    parser.add_argument(
        "--max_speakers",
        type=int,
        help="Maximum number of speakers (optional)",
        default=None
    )
    parser.add_argument(
        "--samples_per_speaker",
        type=int,
        help="Number of audio samples to extract per speaker",
        default=3
    )
    parser.add_argument(
        "--convert_audio",
        action="store_true",
        help="Explicitly convert audio to WAV format before processing"
    )
    args = parser.parse_args()

    # Optionally convert audio to WAV format for better compatibility
    input_file = args.audio_file
    if args.convert_audio or not input_file.lower().endswith(('.wav', '.flac')):
        input_file = convert_to_wav(args.audio_file)

    # Load Whisper model (this will use GPU if available)
    print("Loading Whisper model...")
    model = whisper.load_model(args.whisper_model)

    # Perform speaker diarization
    speaker_segments = perform_diarization(
        input_file, 
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers
    )
    
    # Sort segments by start time
    speaker_segments.sort(key=lambda x: x[0])
    print(f"Diarization complete. Found {len(speaker_segments)} segments.")
    
    # Extract representative samples for each speaker
    speaker_samples = extract_speaker_samples(
        input_file, 
        speaker_segments, 
        samples_per_speaker=args.samples_per_speaker
    )
    
    # Identify speakers through user interaction
    speaker_names = identify_speakers(speaker_samples)

    # Load audio file for transcription
    print("Loading audio for transcription...")
    try:
        # Use librosa which has better format support
        audio_data, sample_rate = librosa.load(input_file, sr=None)
        
        # Process each speaker segment and transcribe it
        transcript_lines = []
        for start, end, speaker_label in speaker_segments:
            speaker_name = speaker_names.get(speaker_label, f"Speaker {speaker_label}")
            
            # Add progress information for long segments
            duration = end - start
            if duration > 20:
                print(f"Transcribing segment {start:.2f}s - {end:.2f}s for {speaker_name} (this might take a while)...")
            else:
                print(f"Transcribing segment {start:.2f}s - {end:.2f}s for {speaker_name}...")
                
            text = transcribe_segment(audio_data, sample_rate, start, end, model)
            transcript_lines.append(f"{speaker_name}: {text}")

        # Combine all transcript lines into a final transcript
        final_transcript = "\n".join(transcript_lines)

        # Write the transcript to the output file
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(final_transcript)

        print(f"Transcript written to {args.output}")
        
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        print("Please make sure your audio file is in a supported format.")
        sys.exit(1)

if __name__ == "__main__":
    main()