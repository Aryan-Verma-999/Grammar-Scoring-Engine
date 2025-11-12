"""
Audio Transcription Script using Faster-Whisper
================================================

This script transcribes audio files from a CSV dataset using the faster-whisper
implementation of OpenAI's Whisper model.
"""

import os
import gc
import argparse
from pathlib import Path
import pandas as pd
import torch
from faster_whisper import WhisperModel


def transcribe_audio_files(df, audio_dir, cache_path=None, model_size="large-v3", 
                           beam_size=5, language="en", verbose=True):
    """
    Transcribe audio files using faster-whisper
    """
    # Check if cached transcripts exist
    if cache_path and os.path.exists(cache_path):
        if verbose:
            print(f"✓ Loading cached transcripts from {cache_path}")
        cached = pd.read_csv(cache_path)
        return cached
    
    # Initialize Whisper model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "float32"
    
    if verbose:
        print(f"Initializing Whisper model ({model_size})...")
        print(f"  Device: {device}")
        print(f"  Compute type: {compute_type}")
    
    whisper = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type
    )
    
    # Transcribe audio files
    transcripts = []
    failed_files = []
    
    if verbose:
        print(f"\nTranscribing {len(df)} audio files...")
        print(f"  Audio directory: {audio_dir}")
        print(f"  Beam size: {beam_size}")
        print(f"  Language: {language}")
        print("-" * 70)
    
    for idx, row in df.iterrows():
        audio_path = Path(audio_dir) / f"{row['filename']}.wav"
        
        try:
            # Check if file exists
            if not audio_path.exists():
                if verbose:
                    print(f"  ⚠ Warning: File not found - {audio_path}")
                transcripts.append("")
                failed_files.append(row['filename'])
                continue
            
            # Transcribe audio
            segments, info = whisper.transcribe(
                str(audio_path), 
                beam_size=beam_size, 
                language=language
            )
            
            # Combine all segments
            text = " ".join([seg.text for seg in segments])
            transcripts.append(text.strip())
            
            # Progress update
            if verbose and (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(df)} files")
        
        except Exception as e:
            if verbose:
                print(f"  ✗ Error processing {row['filename']}: {e}")
            transcripts.append("")
            failed_files.append(row['filename'])
    
    # Add transcripts to dataframe
    df['transcript'] = transcripts
    
    # Print summary
    if verbose:
        print("-" * 70)
        print(f"✓ Transcription complete!")
        print(f"  Successfully transcribed: {len(df) - len(failed_files)}/{len(df)}")
        if failed_files:
            print(f"  Failed files: {len(failed_files)}")
            print(f"    {', '.join(failed_files[:5])}" + 
                  (f" ... (+{len(failed_files)-5} more)" if len(failed_files) > 5 else ""))
    
    # Save to cache if specified
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path, index=False)
        if verbose:
            print(f"✓ Cached transcripts to {cache_path}")
    
    # Cleanup
    del whisper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return df


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using faster-whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        '--csv_path',
        type=str,
        required=True,
        help='Path to CSV file with filename column'
    )
    parser.add_argument(
        '--audio_dir',
        type=str,
        required=True,
        help='Directory containing audio files (.wav)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path to save transcripts (default: <csv_path>_transcripts.csv)'
    )
    parser.add_argument(
        '--model_size',
        type=str,
        default='large-v3',
        choices=['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'],
        help='Whisper model size (default: large-v3)'
    )
    parser.add_argument(
        '--beam_size',
        type=int,
        default=5,
        help='Beam size for decoding (default: 5)'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='en',
        help='Language code for transcription (default: en)'
    )
    parser.add_argument(
        '--no_cache',
        action='store_true',
        help='Disable caching (always transcribe from scratch)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found - {args.csv_path}")
        return
    
    if not os.path.exists(args.audio_dir):
        print(f"Error: Audio directory not found - {args.audio_dir}")
        return
    
    # Set output path
    if args.output_path is None:
        base_name = os.path.splitext(args.csv_path)[0]
        args.output_path = f"{base_name}_transcripts.csv"
    
    # Determine cache path
    cache_path = None if args.no_cache else args.output_path
    
    # Load CSV
    print(f"Loading CSV from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)
    
    if 'filename' not in df.columns:
        print("Error: CSV must contain 'filename' column")
        return
    
    print(f"Found {len(df)} files to transcribe")
    
    # Transcribe
    df_with_transcripts = transcribe_audio_files(
        df=df,
        audio_dir=args.audio_dir,
        cache_path=cache_path,
        model_size=args.model_size,
        beam_size=args.beam_size,
        language=args.language,
        verbose=not args.quiet
    )
    
    # Save results
    if not args.no_cache or args.output_path != cache_path:
        df_with_transcripts.to_csv(args.output_path, index=False)
        print(f"\n✓ Saved transcripts to {args.output_path}")
    
    # Display sample
    print("\nSample transcripts:")
    print("-" * 70)
    for idx in range(min(3, len(df_with_transcripts))):
        filename = df_with_transcripts.iloc[idx]['filename']
        transcript = df_with_transcripts.iloc[idx]['transcript']
        print(f"{filename}:")
        print(f"  {transcript[:100]}{'...' if len(transcript) > 100 else ''}")
        print()


if __name__ == "__main__":
    main()
