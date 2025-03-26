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
    python transcribe.py path/to/audio_file --output transcript.txt
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
import threading
import json
import time
from pathlib import Path
from queue import Queue
from speechbrain.inference.speaker import SpeakerRecognition
import torchaudio
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

class TranscriptionManager:
    """Manages the overall transcription process."""
    
    def __init__(self, args):
        """Initialize with command line arguments."""
        self.args = args
        self.input_file = args.audio_file
        self.output_file = args.output
        self.temp_dir = tempfile.mkdtemp()
        
        # Convert audio if needed
        if args.convert_audio or not self.input_file.lower().endswith(('.wav', '.flac')):
            self.input_file = self._convert_to_wav(self.input_file)
        
        # Initialize components
        self.print_lock = threading.Lock()
        self.speaker_segments = []
        self.speaker_names = {}
        self.transcription_results = []
        
        # Temporary files for intermediate results
        self.temp_segments_file = os.path.join(self.temp_dir, "speaker_segments.json")
        self.temp_names_file = os.path.join(self.temp_dir, "speaker_names.json")
        self.temp_transcription_file = os.path.join(self.temp_dir, "transcription.json")
    
    def _convert_to_wav(self, input_file):
        """Convert any audio file to WAV format for better compatibility."""
        print(f"Converting {input_file} to WAV format for better compatibility...")
        try:
            output_file = os.path.join(self.temp_dir, "converted_audio.wav")
            
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
    
    def synchronized_print(self, message):
        """Thread-safe print function."""
        with self.print_lock:
            print(message)
    
    def run(self):
        """Run the complete transcription process."""
        try:
            # Load the Whisper model
            print("Loading Whisper model...")
            model = whisper.load_model(self.args.whisper_model)
            
            # Run diarization
            self._run_diarization()
            
            # Load audio for transcription
            print("Loading audio for transcription...")
            audio_data, sample_rate = librosa.load(self.input_file, sr=None)
            
            # Extract speaker samples
            speaker_sampler = SpeakerSampler(self.input_file, self.speaker_segments, 
                                             self.args.samples_per_speaker)
            speaker_samples = speaker_sampler.extract_samples()
            
            # Create queues and start transcription
            transcription_queue = Queue()
            progress_queue = Queue()
            
            # Start transcription in a separate thread
            transcriber = Transcriber(audio_data, sample_rate, self.speaker_segments, model)
            transcription_thread = threading.Thread(
                target=transcriber.transcribe_all,
                args=(transcription_queue, progress_queue)
            )
            transcription_thread.daemon = True
            transcription_thread.start()
            
            # Start progress printer in a separate thread
            progress_printer = threading.Thread(
                target=self._print_progress,
                args=(progress_queue,)
            )
            progress_printer.daemon = True
            progress_printer.start()
            
            # Now the user can choose to identify speakers or let transcription complete first
            print("\n" + "="*60)
            print("TRANSCRIPTION STARTED".center(60))
            print("="*60)
            print("Transcription has started in the background.")
            print("You can now identify speakers or wait for transcription to complete.")
            print("1. Identify speakers now")
            print("2. Wait for transcription to complete, then identify speakers")
            print("3. Skip speaker identification (use default speaker labels)")
            
            choice = ""
            while choice not in ["1", "2", "3"]:
                choice = input("Enter your choice (1, 2, or 3): ").strip()
            
            if choice == "1":
                # Identify speakers now
                identifier = SpeakerIdentifier(speaker_samples)
                self.speaker_names = identifier.identify_speakers()
            elif choice == "2":
                # Wait for transcription to complete first
                print("Waiting for transcription to complete before speaker identification...")
                transcription_thread.join()
                progress_printer.join()
                
                # Get transcription results
                while not transcription_queue.empty():
                    speaker_label, text = transcription_queue.get()
                    self.transcription_results.append((speaker_label, text))
                
                # Now identify speakers
                identifier = SpeakerIdentifier(speaker_samples)
                self.speaker_names = identifier.identify_speakers()
            else:
                # Skip identification, use default names
                print("Using default speaker labels...")
                unique_speakers = set(label for _, _, label in self.speaker_segments)
                self.speaker_names = {label: f"Speaker {i+1}" for i, label in enumerate(unique_speakers)}
            
            # If we haven't collected results yet, wait for transcription now
            if not self.transcription_results:
                if not transcription_thread.is_alive():
                    # Thread completed while user was making a choice
                    while not transcription_queue.empty():
                        speaker_label, text = transcription_queue.get()
                        self.transcription_results.append((speaker_label, text))
                else:
                    # Wait for transcription to complete
                    print("Waiting for transcription to complete...")
                    transcription_thread.join()
                    progress_printer.join()
                    
                    # Get transcription results
                    while not transcription_queue.empty():
                        speaker_label, text = transcription_queue.get()
                        self.transcription_results.append((speaker_label, text))
            
            # Generate final transcript
            self._generate_transcript()
            
            # Clean up temporary directory
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            
            print(f"Transcript written to {self.output_file}")
            
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            print("Please make sure your audio file is in a supported format.")
            sys.exit(1)
    
    def _run_diarization(self):
        """Run speaker diarization on the audio file."""
        # Check if we have cached diarization results
        if os.path.exists(self.temp_segments_file):
            print("Loading cached diarization results...")
            with open(self.temp_segments_file, 'r') as f:
                self.speaker_segments = json.load(f)
            return
        
        print("Performing speaker diarization...")
        diarizer = Diarizer(self.input_file, self.args.min_speakers, self.args.max_speakers)
        self.speaker_segments = diarizer.diarize()
        
        # Sort segments by start time
        self.speaker_segments.sort(key=lambda x: x[0])
        print(f"Diarization complete. Found {len(self.speaker_segments)} segments.")
        
        # Save segments for potential reuse
        with open(self.temp_segments_file, 'w') as f:
            json.dump(self.speaker_segments, f)
    
    def _print_progress(self, progress_queue):
        """Print transcription progress from queue."""
        print("\nTranscription Progress:")
        print("-" * 50)
        
        while True:
            message = progress_queue.get()
            if message is None:  # End signal
                break
            self.synchronized_print(message)
    
    def _generate_transcript(self):
        """Generate the final transcript with speaker names."""
        # Sort results by segment start time
        segment_starts = {label: start for start, _, label in self.speaker_segments}
        self.transcription_results.sort(key=lambda x: segment_starts.get(x[0], 0))
        
        # Format with speaker names
        transcript_lines = []
        for speaker_label, text in self.transcription_results:
            speaker_name = self.speaker_names.get(speaker_label, f"Speaker {speaker_label}")
            transcript_lines.append(f"{speaker_name}: {text}")
        
        # Write to output file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(transcript_lines))


class Diarizer:
    """Handles speaker diarization."""
    
    def __init__(self, audio_file, min_speakers=None, max_speakers=None):
        """Initialize with audio file and speaker range."""
        self.audio_file = audio_file
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
    
    def diarize(self):
        """
        Perform speaker diarization on an audio file.
        
        Returns:
            list: List of (start_time, end_time, speaker_id) tuples
        """
        try:
            # Load the SpeechBrain speaker recognition model
            speaker_model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb", 
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )
            
            # Load the audio file
            signal, sample_rate = torchaudio.load(self.audio_file)
            
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
            
            if self.min_speakers is not None and self.max_speakers is not None:
                if self.min_speakers == self.max_speakers:
                    num_speakers = self.min_speakers
                else:
                    # Use agglomerative clustering to estimate number of speakers
                    clustering = AgglomerativeClustering(
                        n_clusters=None,
                        distance_threshold=0.7,  # Adjust as needed
                        affinity='cosine',
                        linkage='average'
                    ).fit(embeddings)
                    num_speakers = len(set(clustering.labels_))
                    num_speakers = max(min(num_speakers, self.max_speakers), self.min_speakers)
            elif self.min_speakers is not None:
                num_speakers = max(2, self.min_speakers)
            elif self.max_speakers is not None:
                num_speakers = min(self.max_speakers, 8)  # Default upper limit
                
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
                duration = librosa.get_duration(path=self.audio_file)
            except:
                # If that fails, try a more robust approach
                info = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", self.audio_file], 
                                   capture_output=True, text=True)
                duration = float(info.stdout.strip())
                
            return [(0.0, duration, "spk_0")]


class SpeakerSampler:
    """Handles extracting representative audio samples for each speaker."""
    
    def __init__(self, audio_file, speaker_segments, samples_per_speaker=3, max_sample_length=5):
        """Initialize with audio file and speaker segments."""
        self.audio_file = audio_file
        self.speaker_segments = speaker_segments
        self.samples_per_speaker = samples_per_speaker
        self.max_sample_length = max_sample_length
        self.temp_dir = tempfile.mkdtemp()
    
    def extract_samples(self):
        """
        Extract representative audio samples for each speaker.
        
        Returns:
            dict: Mapping of speaker_id to list of sample file paths
        """
        print("Extracting representative samples for each speaker...")
        
        # Load the audio file
        y, sr = librosa.load(self.audio_file, sr=None)
        
        # Group segments by speaker
        speaker_to_segments = {}
        for start, end, speaker_id in self.speaker_segments:
            if speaker_id not in speaker_to_segments:
                speaker_to_segments[speaker_id] = []
            speaker_to_segments[speaker_id].append((start, end))
        
        # Extract multiple segments for each speaker
        speaker_samples = {}
        for speaker_id, segments in speaker_to_segments.items():
            # Sort segments by duration (longest first)
            segments.sort(key=lambda x: x[1] - x[0], reverse=True)
            
            # Get up to samples_per_speaker segments
            speaker_samples[speaker_id] = []
            for i, (start, end) in enumerate(segments[:self.samples_per_speaker]):
                # Limit to max_sample_length
                if end - start > self.max_sample_length:
                    end = start + self.max_sample_length
                
                # Extract the audio segment
                start_idx = int(start * sr)
                end_idx = int(end * sr)
                segment_audio = y[start_idx:end_idx]
                
                # Save to a temporary WAV file
                sample_path = os.path.join(self.temp_dir, f"{speaker_id}_sample{i+1}.wav")
                sf.write(sample_path, segment_audio, sr)
                
                speaker_samples[speaker_id].append(sample_path)
        
        return speaker_samples


class SpeakerIdentifier:
    """Handles identification of speakers through user interaction."""
    
    def __init__(self, speaker_samples):
        """Initialize with speaker samples."""
        self.speaker_samples = speaker_samples
        self.print_lock = threading.Lock()
    
    def play_audio_sample(self, audio_file):
        """Play an audio sample using pygame."""
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
    
    def identify_speakers(self):
        """
        Play sample audio for each speaker and prompt user for names.
        
        Returns:
            dict: Mapping of speaker_id to speaker name
        """
        with self.print_lock:
            print("\n" + "="*60)
            print("SPEAKER IDENTIFICATION".center(60))
            print("="*60)
            print("I'll play a sample of each speaker's voice.")
            print("Please provide a name for each speaker.")
            print("If the sample is unclear, just press Enter and I'll play another sample.")
            print("-"*60)
        
        input("Press Enter when you're ready to begin speaker identification...")
        
        speaker_names = {}
        
        for i, (speaker_id, sample_paths) in enumerate(self.speaker_samples.items(), 1):
            default_name = f"Speaker {i}"
            
            # Try each sample until user provides a name or we run out of samples
            for j, sample_path in enumerate(sample_paths, 1):
                with self.print_lock:
                    print(f"\nSpeaker {i} sample {j}/{len(sample_paths)}:")
                
                # Play the sample
                self.play_audio_sample(sample_path)
                
                # Ask for speaker name
                name = input(f"Enter name for {default_name} (or press Enter for another sample): ").strip()
                
                # If name provided, use it and move to next speaker
                if name:
                    speaker_names[speaker_id] = name
                    break
                    
                # If this was the last sample, use default name
                if j == len(sample_paths):
                    with self.print_lock:
                        print(f"No more samples available. Using default name '{default_name}'.")
                    speaker_names[speaker_id] = default_name
                    break
                
                # Otherwise, continue to next sample
                with self.print_lock:
                    print("Trying another sample...")
            
        with self.print_lock:
            print("\nSpeaker identification complete!")
            print("-"*60)
        
        return speaker_names


class Transcriber:
    """Handles transcription of audio segments."""
    
    def __init__(self, audio_data, sample_rate, speaker_segments, model):
        """Initialize with audio data and segments."""
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.speaker_segments = speaker_segments
        self.model = model
    
    def transcribe_segment(self, start, end):
        """
        Transcribe a segment of the audio using Whisper.
        
        Parameters:
          start (float): Start time in seconds.
          end (float): End time in seconds.
          
        Returns:
          str: Transcribed text.
        """
        # Convert times to sample indices
        start_idx = int(start * self.sample_rate)
        end_idx = int(end * self.sample_rate)
        segment_audio = self.audio_data[start_idx:end_idx]

        # Convert multi-channel audio to mono if necessary
        if segment_audio.ndim > 1:
            segment_audio = np.mean(segment_audio, axis=1)

        # Resample segment if sample rate is not 16kHz
        target_sr = 16000
        if self.sample_rate != target_sr:
            segment_audio = librosa.resample(segment_audio, orig_sr=self.sample_rate, target_sr=target_sr)
            sr_for_whisper = target_sr
        else:
            sr_for_whisper = self.sample_rate

        # Ensure the audio is in float32
        segment_audio = segment_audio.astype(np.float32)

        # Transcribe with Whisper (forcing English for consistency)
        result = self.model.transcribe(segment_audio, fp16=False, language='en')
        return result.get("text", "").strip()
    
    def transcribe_all(self, transcription_queue, progress_queue):
        """
        Transcribe all segments and put results in queues.
        
        Parameters:
            transcription_queue (Queue): Queue to store transcription results
            progress_queue (Queue): Queue to store progress updates
        """
        total_segments = len(self.speaker_segments)
        
        for i, (start, end, speaker_label) in enumerate(self.speaker_segments):
            duration = end - start
            if duration > 20:
                progress_queue.put(f"[{i+1}/{total_segments}] Transcribing segment {start:.2f}s - {end:.2f}s (this might take a while)...")
            else:
                progress_queue.put(f"[{i+1}/{total_segments}] Transcribing segment {start:.2f}s - {end:.2f}s...")
                
            text = self.transcribe_segment(start, end)
            transcription_queue.put((speaker_label, text))
        
        # Signal completion
        progress_queue.put("Transcription complete!")
        progress_queue.put(None)


class TranscriptionApp:
    """GUI application for transcription with speaker identification."""
    
    def __init__(self, root):
        """Initialize the application with root window."""
        self.root = root
        self.root.title("Audio Transcription Tool")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)
        
        self.temp_dir = tempfile.mkdtemp()
        self.audio_file = None
        self.output_file = None
        self.speaker_samples = None
        self.speaker_names = {}
        
        # Create main frames
        self.create_frames()
        
        # Create widgets
        self.create_input_widgets()
        self.create_output_widgets()
        
        # Set up redirected stdout for logging
        self.setup_logging()
        
        # UI state variables
        self.is_transcribing = False
        self.transcription_thread = None
        
        # Initialize variables for speaker identification
        self.current_speaker_id = None
        self.current_speaker_idx = 0
        self.current_sample_idx = 0
        self.speaker_ids = []
        self.speaker_sample_paths = {}
    
    def create_frames(self):
        """Create the main frames for the UI."""
        # Top frame for input parameters
        self.input_frame = ttk.LabelFrame(self.root, text="Input Parameters")
        self.input_frame.pack(fill=tk.X, padx=10, pady=5, expand=False)
        
        # Frame for action buttons
        self.action_frame = ttk.Frame(self.root)
        self.action_frame.pack(fill=tk.X, padx=10, pady=5, expand=False)
        
        # Frame for output and logs
        self.output_frame = ttk.LabelFrame(self.root, text="Output")
        self.output_frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)
        
        # Frame for speaker identification (initially hidden)
        self.speaker_id_frame = ttk.LabelFrame(self.root, text="Speaker Identification")
        
    def create_input_widgets(self):
        """Create widgets for input parameters."""
        # Audio file selection
        ttk.Label(self.input_frame, text="Audio File:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.audio_path_var = tk.StringVar()
        ttk.Entry(self.input_frame, textvariable=self.audio_path_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.input_frame, text="Browse...", command=self.browse_audio_file).grid(row=0, column=2, padx=5, pady=5)
        
        # Output file selection
        ttk.Label(self.input_frame, text="Output File:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.output_path_var = tk.StringVar(value="transcript.txt")
        ttk.Entry(self.input_frame, textvariable=self.output_path_var, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(self.input_frame, text="Browse...", command=self.browse_output_file).grid(row=1, column=2, padx=5, pady=5)
        
        # Whisper model selection
        ttk.Label(self.input_frame, text="Whisper Model:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_var = tk.StringVar(value="small")
        models = ["tiny", "base", "small", "medium", "large"]
        ttk.Combobox(self.input_frame, textvariable=self.model_var, values=models, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Number of speakers
        frame = ttk.Frame(self.input_frame)
        frame.grid(row=3, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(frame, text="Min Speakers:").pack(side=tk.LEFT, padx=5)
        self.min_speakers_var = tk.IntVar(value=0)
        ttk.Spinbox(frame, from_=0, to=10, textvariable=self.min_speakers_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(frame, text="Max Speakers:").pack(side=tk.LEFT, padx=5)
        self.max_speakers_var = tk.IntVar(value=0)
        ttk.Spinbox(frame, from_=0, to=10, textvariable=self.max_speakers_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # Convert audio checkbox
        self.convert_audio_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.input_frame, text="Convert audio to WAV format", variable=self.convert_audio_var).grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # Action buttons
        ttk.Button(self.action_frame, text="Start Transcription", command=self.start_transcription).pack(side=tk.LEFT, padx=5, pady=5)
        self.cancel_button = ttk.Button(self.action_frame, text="Cancel", command=self.cancel_transcription, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, padx=5, pady=5)
    
    def create_output_widgets(self):
        """Create widgets for output and logs."""
        # Progress bar
        self.progress_frame = ttk.Frame(self.output_frame)
        self.progress_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(self.progress_frame, text="Progress:").pack(side=tk.LEFT, padx=5)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.progress_label = ttk.Label(self.progress_frame, text="0%")
        self.progress_label.pack(side=tk.LEFT, padx=5)
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(self.output_frame, height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)
    
    def setup_logging(self):
        """Redirect stdout to the log text widget."""
        class TextRedirector:
            def __init__(self, widget):
                self.widget = widget
                self.buffer = ""
                
            def write(self, string):
                self.buffer += string
                if '\n' in self.buffer:
                    lines = self.buffer.split('\n')
                    self.buffer = lines[-1]
                    for line in lines[:-1]:
                        self.widget.config(state=tk.NORMAL)
                        self.widget.insert(tk.END, line + '\n')
                        self.widget.see(tk.END)
                        self.widget.config(state=tk.DISABLED)
                        
            def flush(self):
                if self.buffer:
                    self.widget.config(state=tk.NORMAL)
                    self.widget.insert(tk.END, self.buffer)
                    self.widget.see(tk.END)
                    self.widget.config(state=tk.DISABLED)
                    self.buffer = ""
        
        sys.stdout = TextRedirector(self.log_text)
    
    def browse_audio_file(self):
        """Open file dialog to select audio file."""
        filetypes = [
            ("Audio files", "*.wav *.mp3 *.flac *.m4a *.ogg"),
            ("All files", "*.*")
        ]
        filename = filedialog.askopenfilename(filetypes=filetypes, title="Select Audio File")
        if filename:
            self.audio_path_var.set(filename)
            # Set default output file name based on input file
            if self.output_path_var.get() == "transcript.txt":
                input_path = Path(filename)
                self.output_path_var.set(str(input_path.with_suffix('.txt')))
    
    def browse_output_file(self):
        """Open file dialog to select output file."""
        filetypes = [
            ("Text files", "*.txt"),
            ("All files", "*.*")
        ]
        filename = filedialog.asksaveasfilename(filetypes=filetypes, defaultextension=".txt", title="Save Transcript As")
        if filename:
            self.output_path_var.set(filename)
    
    def start_transcription(self):
        """Start the transcription process."""
        # Validate inputs
        self.audio_file = self.audio_path_var.get()
        self.output_file = self.output_path_var.get()
        
        if not self.audio_file:
            messagebox.showerror("Error", "Please select an audio file")
            return
        
        if not os.path.exists(self.audio_file):
            messagebox.showerror("Error", "Audio file does not exist")
            return
        
        # Prepare arguments
        min_speakers = self.min_speakers_var.get() if self.min_speakers_var.get() > 0 else None
        max_speakers = self.max_speakers_var.get() if self.max_speakers_var.get() > 0 else None
        
        # Create a custom args object
        class Args:
            pass
        
        args = Args()
        args.audio_file = self.audio_file
        args.output = self.output_file
        args.whisper_model = self.model_var.get()
        args.min_speakers = min_speakers
        args.max_speakers = max_speakers
        args.samples_per_speaker = 3
        args.convert_audio = self.convert_audio_var.get()
        
        # Disable UI during transcription
        self.toggle_ui_state(False)
        
        # Start transcription in a background thread
        self.transcription_thread = threading.Thread(target=self.run_transcription, args=(args,))
        self.transcription_thread.daemon = True
        self.transcription_thread.start()
    
    def run_transcription(self, args):
        """Run transcription in a background thread."""
        try:
            # Initialize the transcription manager
            manager = TranscriptionManager(args)
            
            # Override methods to use GUI
            manager.original_run = manager.run
            manager.run = lambda: None  # Prevent normal run
            
            # Load the Whisper model
            print("Loading Whisper model...")
            model = whisper.load_model(args.whisper_model)
            
            # Run diarization
            manager._run_diarization()
            
            # Load audio for transcription
            print("Loading audio for transcription...")
            audio_data, sample_rate = librosa.load(manager.input_file, sr=None)
            
            # Extract speaker samples
            speaker_sampler = SpeakerSampler(manager.input_file, manager.speaker_segments, 
                                             args.samples_per_speaker)
            self.speaker_samples = speaker_sampler.extract_samples()
            
            # Create queues for transcription
            transcription_queue = Queue()
            progress_queue = Queue()
            
            # Start transcription in a separate thread
            transcriber = Transcriber(audio_data, sample_rate, manager.speaker_segments, model)
            
            # Override transcribe_all to update progress
            original_transcribe_all = transcriber.transcribe_all
            def transcribe_all_with_progress(tq, pq):
                total_segments = len(transcriber.speaker_segments)
                
                for i, (start, end, speaker_label) in enumerate(transcriber.speaker_segments):
                    # Update progress
                    progress = (i / total_segments) * 100
                    self.root.after(0, self.update_progress, progress)
                    
                    duration = end - start
                    if duration > 20:
                        pq.put(f"[{i+1}/{total_segments}] Transcribing segment {start:.2f}s - {end:.2f}s (this might take a while)...")
                    else:
                        pq.put(f"[{i+1}/{total_segments}] Transcribing segment {start:.2f}s - {end:.2f}s...")
                        
                    text = transcriber.transcribe_segment(start, end)
                    tq.put((speaker_label, text))
                
                # Signal completion
                pq.put("Transcription complete!")
                pq.put(None)
                
                # Update progress to 100%
                self.root.after(0, self.update_progress, 100)
                
                # Show speaker identification dialog
                self.root.after(0, self.show_speaker_identification)
            
            transcriber.transcribe_all = transcribe_all_with_progress
            
            # Start transcription
            print("Starting transcription...")
            transcriber.transcribe_all(transcription_queue, progress_queue)
            
            # Process transcription results
            manager.transcription_results = []
            while not transcription_queue.empty():
                speaker_label, text = transcription_queue.get()
                manager.transcription_results.append((speaker_label, text))
            
            # Speaker identification is handled by UI
            
            # Generate and save transcript using speaker names
            manager.speaker_names = self.speaker_names
            manager._generate_transcript()
            
            # Clean up
            print(f"Transcript written to {manager.output_file}")
            
            # Enable UI after completion
            self.root.after(0, self.toggle_ui_state, True)
            
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            # Enable UI after error
            self.root.after(0, self.toggle_ui_state, True)
            self.root.after(0, lambda: messagebox.showerror("Error", f"Transcription failed: {str(e)}"))
    
    def update_progress(self, value):
        """Update the progress bar."""
        self.progress_var.set(value)
        self.progress_label.config(text=f"{int(value)}%")
    
    def toggle_ui_state(self, enabled):
        """Enable or disable UI elements during processing."""
        state = tk.NORMAL if enabled else tk.DISABLED
        for child in self.input_frame.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                pass
        
        for child in self.action_frame.winfo_children():
            if child != self.cancel_button:
                try:
                    child.configure(state=state)
                except tk.TclError:
                    pass
        
        self.cancel_button.configure(state=tk.DISABLED if enabled else tk.NORMAL)
        self.is_transcribing = not enabled
    
    def cancel_transcription(self):
        """Cancel ongoing transcription."""
        if self.is_transcribing and self.transcription_thread and self.transcription_thread.is_alive():
            # This doesn't actually stop the thread, but flags the UI as available
            self.toggle_ui_state(True)
            messagebox.showinfo("Cancelled", "Transcription process cancelled. Thread will terminate when current task completes.")
    
    def show_speaker_identification(self):
        """Show the speaker identification dialog."""
        # Create speaker identification window
        self.speaker_window = tk.Toplevel(self.root)
        self.speaker_window.title("Speaker Identification")
        self.speaker_window.geometry("500x300")
        self.speaker_window.transient(self.root)
        self.speaker_window.grab_set()
        
        # Setup widgets
        ttk.Label(self.speaker_window, text="Listen to each speaker sample and provide a name.").pack(pady=10)
        
        # Speaker info frame
        info_frame = ttk.Frame(self.speaker_window)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.speaker_label = ttk.Label(info_frame, text="Speaker 1")
        self.speaker_label.pack(side=tk.LEFT, padx=5)
        
        self.sample_label = ttk.Label(info_frame, text="Sample 1/3")
        self.sample_label.pack(side=tk.RIGHT, padx=5)
        
        # Play button
        ttk.Button(self.speaker_window, text="Play Sample", command=self.play_current_sample).pack(pady=10)
        
        # Input frame
        input_frame = ttk.Frame(self.speaker_window)
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(input_frame, text="Speaker Name:").pack(side=tk.LEFT, padx=5)
        self.name_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.name_var, width=30).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Button frame
        button_frame = ttk.Frame(self.speaker_window)
        button_frame.pack(fill=tk.X, padx=10, pady=15)
        
        ttk.Button(button_frame, text="Next Sample", command=self.next_sample).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Next Speaker", command=self.next_speaker).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Use Default Names", command=self.use_default_names).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Finish", command=self.finish_identification).pack(side=tk.RIGHT, padx=5)
        
        # Initialize speaker id sequence
        self.speaker_ids = list(self.speaker_samples.keys())
        self.current_speaker_idx = 0
        self.current_sample_idx = 0
        
        if self.speaker_ids:
            self.current_speaker_id = self.speaker_ids[0]
            self.update_speaker_ui()
    
    def play_current_sample(self):
        """Play the current speaker sample."""
        if not self.speaker_ids:
            return
            
        current_samples = self.speaker_samples[self.current_speaker_id]
        if self.current_sample_idx < len(current_samples):
            sample_path = current_samples[self.current_sample_idx]
            
            # Play in a separate thread to avoid freezing UI
            threading.Thread(target=self.play_audio, args=(sample_path,)).start()
    
    def play_audio(self, audio_file):
        """Play audio in a separate thread."""
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
            pygame.mixer.quit()
        except Exception as e:
            print(f"Error playing audio: {e}")
    
    def next_sample(self):
        """Move to the next sample for the current speaker."""
        current_samples = self.speaker_samples[self.current_speaker_id]
        self.current_sample_idx = (self.current_sample_idx + 1) % len(current_samples)
        self.update_speaker_ui()
    
    def next_speaker(self):
        """Save current speaker name and move to the next speaker."""
        # Save current speaker name if provided
        name = self.name_var.get().strip()
        if name:
            self.speaker_names[self.current_speaker_id] = name
        
        # Move to next speaker
        if self.current_speaker_idx < len(self.speaker_ids) - 1:
            self.current_speaker_idx += 1
            self.current_speaker_id = self.speaker_ids[self.current_speaker_idx]
            self.current_sample_idx = 0
            self.name_var.set("")
            self.update_speaker_ui()
        else:
            # All speakers processed
            self.finish_identification()
    
    def update_speaker_ui(self):
        """Update the speaker identification UI."""
        if not self.speaker_ids:
            return
            
        # Update labels
        self.speaker_label.config(text=f"Speaker {self.current_speaker_idx + 1}/{len(self.speaker_ids)}")
        
        current_samples = self.speaker_samples[self.current_speaker_id]
        self.sample_label.config(text=f"Sample {self.current_sample_idx + 1}/{len(current_samples)}")
    
    def use_default_names(self):
        """Use default speaker names instead of manual identification."""
        for i, speaker_id in enumerate(self.speaker_ids):
            self.speaker_names[speaker_id] = f"Speaker {i+1}"
        
        self.finish_identification()
    
    def finish_identification(self):
        """Complete speaker identification process."""
        # Ensure all speakers have names
        for i, speaker_id in enumerate(self.speaker_ids):
            if speaker_id not in self.speaker_names:
                self.speaker_names[speaker_id] = f"Speaker {i+1}"
        
        # Close the speaker window
        if hasattr(self, 'speaker_window'):
            self.speaker_window.destroy()
        
        print("Speaker identification complete!")
        messagebox.showinfo("Complete", "Transcription and speaker identification complete!")

def main():
    """Main entry point for the application."""
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()