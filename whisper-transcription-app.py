import streamlit as st
import whisper
import tempfile
import os
from datetime import timedelta
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Video Transcript Generator",
    page_icon="🎬",
    layout="wide"
)

# Application title and description
st.title("Video Transcript Generator")
st.markdown("""
This application uses OpenAI's Whisper model to transcribe audio from video files.
Upload your video, select a Whisper model, and get accurate transcriptions with timestamps.
""")

# Sidebar for model selection and options
with st.sidebar:
    st.header("Model Settings")
    
    model_size = st.selectbox(
        "Select Whisper Model",
        options=["tiny", "base", "small", "medium", "large"],
        index=1,
        help="Larger models are more accurate but slower and require more memory"
    )
    
    language = st.selectbox(
        "Select Language (optional)",
        options=["", "English", "Spanish", "French", "German", "Italian", "Portuguese", "Dutch"],
        index=0,
        help="Leave blank for automatic language detection"
    )
    
    show_timestamps = st.checkbox("Show Timestamps", value=True)
    
    # Map UI language selection to Whisper language codes
    language_map = {
        "English": "en", 
        "Spanish": "es", 
        "French": "fr", 
        "German": "de", 
        "Italian": "it", 
        "Portuguese": "pt",
        "Dutch": "nl"
    }
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This application utilizes OpenAI's Whisper, an automatic speech recognition system,
    to generate accurate transcriptions of video content.
    """)

# Function to format time
def format_time(seconds):
    return str(timedelta(seconds=seconds))

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
        
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_file_path = tmp_file.name
        
        # Process the video with Whisper
        with st.spinner("Processing video... This may take a while depending on the video length and model size."):
            try:
                # Load the selected Whisper model
                model = whisper.load_model(model_size)
                
                # Set transcription options
                options = {}
                if language and language in language_map:
                    options["language"] = language_map[language]
                
                # Perform the transcription
                result = model.transcribe(temp_file_path, **options)
                
                # Display success message
                st.success("Transcription completed!")
                
                # Display transcription results
                st.header("Transcription Results")
                
                # Full transcript without timestamps
                st.subheader("Full Transcript")
                st.markdown(result["text"])
                
                # Segments with timestamps if selected
                if show_timestamps and "segments" in result:
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
                
                # Create a downloadable SRT file if timestamps are available
                if "segments" in result:
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
            
            # Clean up the temporary file
            os.unlink(temp_file_path)

# Execute the main function
process_video()

# Footer
st.markdown("---")
st.markdown("Powered by OpenAI's Whisper model • Built with Streamlit")
