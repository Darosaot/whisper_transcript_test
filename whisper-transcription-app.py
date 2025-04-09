import streamlit as st
import tempfile
import os
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import pandas as pd
from datetime import timedelta
import subprocess

# Page configuration
st.set_page_config(
    page_title="Video Transcript Generator",
    page_icon="🎬",
    layout="wide"
)

# Application title and description
st.title("Video Transcript Generator")
st.markdown("""
This application transcribes speech from video files to text using SpeechRecognition library.
Upload your video, select options, and get accurate transcriptions.
""")

# Sidebar for options
with st.sidebar:
    st.header("Transcription Settings")
    
    recognition_engine = st.selectbox(
        "Select Recognition Engine",
        options=["Google (Online)", "Sphinx (Offline)"],
        index=0,
        help="Google provides better accuracy but requires internet connection"
    )
    
    language = st.selectbox(
        "Select Language (for Google only)",
        options=["en-US", "es-ES", "fr-FR", "de-DE", "it-IT", "pt-PT", "nl-NL"],
        index=0,
        help="Language selection for Google Speech Recognition"
    )
    
    split_audio = st.checkbox("Split Audio on Silence", value=True, 
                            help="Split audio into smaller chunks based on silence for better results")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This application uses the SpeechRecognition library to transcribe audio from video files.
    It supports both online (Google) and offline (CMU Sphinx) transcription engines.
    """)

# Function to format time
def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

# Function to extract audio from video
def extract_audio_from_video(video_path):
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    command = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path, "-y"]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return audio_path
    except subprocess.CalledProcessError as e:
        st.error(f"Error extracting audio: {e}")
        return None

# Function to transcribe audio
def transcribe_audio(audio_path, engine="google", language="en-US", split_on_silence_flag=True):
    recognizer = sr.Recognizer()
    
    # Load the audio file
    audio = AudioSegment.from_wav(audio_path)
    
    # Initialize results
    full_transcript = ""
    segments = []
    
    if split_on_silence_flag:
        # Split audio on silence
        st.info("Splitting audio on silence...")
        chunks = split_on_silence(
            audio,
            min_silence_len=500,  # minimum silence length in ms
            silence_thresh=audio.dBFS - 14,  # silence threshold
            keep_silence=500  # keep 500ms of leading/trailing silence
        )
        
        # Process each chunk
        with st.progress(0) as progress_bar:
            for i, chunk in enumerate(chunks):
                # Export chunk for processing
                chunk_path = os.path.join(tempfile.gettempdir(), f"chunk_{i}.wav")
                chunk.export(chunk_path, format="wav")
                
                # Load the chunk for recognition
                with sr.AudioFile(chunk_path) as source:
                    audio_data = recognizer.record(source)
                
                # Recognize using selected engine
                try:
                    if engine == "google":
                        text = recognizer.recognize_google(audio_data, language=language)
                    else:  # Use Sphinx
                        text = recognizer.recognize_sphinx(audio_data)
                    
                    # Calculate timestamps
                    start_time = sum(len(c) for c in chunks[:i]) / 1000.0
                    end_time = start_time + len(chunk) / 1000.0
                    
                    # Add to segments
                    segments.append({
                        "start": start_time,
                        "end": end_time,
                        "text": text
                    })
                    
                    # Add to full transcript
                    full_transcript += text + " "
                    
                except sr.UnknownValueError:
                    pass  # Skip segments that couldn't be recognized
                except sr.RequestError as e:
                    st.error(f"Recognition request error: {e}")
                    break
                
                # Update progress
                progress_bar.progress((i + 1) / len(chunks))
                
                # Remove temporary chunk file
                os.remove(chunk_path)
    else:
        # Process the entire audio file at once
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        
        try:
            if engine == "google":
                full_transcript = recognizer.recognize_google(audio_data, language=language)
            else:  # Use Sphinx
                full_transcript = recognizer.recognize_sphinx(audio_data)
                
            # Create a single segment for the entire audio
            segments.append({
                "start": 0,
                "end": len(audio) / 1000.0,
                "text": full_transcript
            })
        except sr.UnknownValueError:
            st.error("Could not understand the audio")
        except sr.RequestError as e:
            st.error(f"Recognition request error: {e}")
    
    return {
        "text": full_transcript.strip(),
        "segments": segments
    }

# Main function to handle file upload and transcription
def process_video():
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        # Display file information
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / (1024*1024):.2f} MB"
        }
        st.write("File Information:")
        for key, value in file_details.items():
            st.write(f"- {key}: {value}")
        
        # Check for FFmpeg
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            st.error("""
            FFmpeg is not installed or not in PATH. This is required for processing video files.
            
            Please install FFmpeg:
            - For Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg
            - For MacOS: brew install ffmpeg
            - For Windows: Download from https://ffmpeg.org/download.html and add to PATH
            """)
            return
            
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            video_path = tmp_file.name
        
        # Transcribe the video
        if st.button("Start Transcription"):
            with st.spinner("Processing video... This may take a while depending on the video length."):
                try:
                    # Step 1: Extract audio from video
                    st.info("Extracting audio from video...")
                    audio_path = extract_audio_from_video(video_path)
                    
                    if not audio_path:
                        st.error("Failed to extract audio from the video file.")
                        return
                    
                    # Step 2: Transcribe the audio
                    st.info("Transcribing audio...")
                    
                    # Get the engine selection
                    engine = "google" if recognition_engine == "Google (Online)" else "sphinx"
                    
                    # Perform transcription
                    result = transcribe_audio(
                        audio_path, 
                        engine=engine,
                        language=language,
                        split_on_silence_flag=split_audio
                    )
                    
                    # Display success message
                    st.success("Transcription completed!")
                    
                    # Display transcription results
                    st.header("Transcription Results")
                    
                    # Full transcript without timestamps
                    st.subheader("Full Transcript")
                    st.markdown(result["text"])
                    
                    # Segments with timestamps
                    if result["segments"]:
                        st.subheader("Segments with Timestamps")
                        
                        # Create a DataFrame for better display
                        segments_data = []
                        for segment in result["segments"]:
                            segments_data.append({
                                "Start": format_time(segment["start"]),
                                "End": format_time(segment["end"]),
                                "Text": segment["text"]
                            })
                        
                        segments_df = pd.DataFrame(segments_data)
                        st.dataframe(segments_df, use_container_width=True)
                    
                    # Download options
                    st.subheader("Download Options")
                    
                    # Prepare the transcript for download
                    full_transcript = result["text"]
                    
                    # Create a downloadable full transcript
                    st.download_button(
                        label="Download Full Transcript (TXT)",
                        data=full_transcript,
                        file_name=f"{uploaded_file.name}_transcript.txt",
                        mime="text/plain"
                    )
                    
                    # Create a downloadable SRT file if segments are available
                    if result["segments"]:
                        srt_content = ""
                        for i, segment in enumerate(result["segments"]):
                            start_time = format_time(segment["start"]).replace(":", ",", 1).replace(":", ",", 1)
                            end_time = format_time(segment["end"]).replace(":", ",", 1).replace(":", ",", 1)
                            srt_content += f"{i+1}\n{start_time} --> {end_time}\n{segment['text']}\n\n"
                        
                        st.download_button(
                            label="Download Subtitles (SRT)",
                            data=srt_content,
                            file_name=f"{uploaded_file.name}_subtitles.srt",
                            mime="text/plain"
                        )
                
                except Exception as e:
                    st.error(f"An error occurred during transcription: {str(e)}")
                
                # Clean up temporary files
                try:
                    os.unlink(video_path)
                    os.unlink(audio_path)
                except:
                    pass  # Ignore errors during cleanup

# Execute the main function
process_video()

# Footer
st.markdown("---")
st.markdown("Powered by SpeechRecognition • Built with Streamlit")
