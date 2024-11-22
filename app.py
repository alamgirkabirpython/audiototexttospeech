import streamlit as st
from transformers import pipeline, VitsModel, AutoTokenizer
from pydub import AudioSegment
import torch
import numpy as np
import os
import time
import yt_dlp
import tempfile

# Set Streamlit page config
st.set_page_config(
    page_title="Audio/Video Transcription & Text-to-Speech",
    layout="centered",
    initial_sidebar_state="auto"
)

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the Whisper model pipeline
@st.cache_resource
def load_transcription_model():
    return pipeline(
        "automatic-speech-recognition",
        "openai/whisper-small",
        chunk_length_s=30,
        stride_length_s=3,
        return_timestamps=True,
        device=device,
    )

transcription_pipe = load_transcription_model()

# Load Text-to-Speech models for English and Hindi
@st.cache_resource
def load_tts_model(language_code):
    model_name = f"facebook/mms-tts-{language_code}"
    return VitsModel.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)

tts_models = {
    "English": load_tts_model("eng"),
    "Hindi": load_tts_model("hin")
}

# Function to download audio using yt-dlp
def download_audio_youtube(video_url, output_path="temp_audio.mp4"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'quiet': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return output_path
    except yt_dlp.utils.DownloadError as e:
        st.error(f"Error downloading audio from YouTube: {e}")
        return None

# Function to convert audio to WAV format using pydub
def convert_to_wav(input_file, output_file="temp_audio.wav"):
    audio = AudioSegment.from_file(input_file)
    audio.export(output_file, format="wav")
    return output_file

# Function to format transcription with timestamps and combine text
def format_transcription(transcription):
    formatted_text, full_text, previous_text = "", "", ""
    for chunk in transcription.get('chunks', []):
        text = chunk["text"]
        timestamps = chunk.get("timestamp", None)

        # Check if timestamps exist
        if timestamps:
            start_time, end_time = timestamps
            formatted_text += f"[{start_time:.2f} - {end_time:.2f}] {text.strip()}\n"
        else:
            formatted_text += f"[No Timestamp] {text.strip()}\n"

        # Avoid duplicate consecutive text
        if text.strip() != previous_text:
            full_text += text.strip() + " "
            previous_text = text.strip()

    return formatted_text.strip(), full_text.strip()

# Function to generate TTS output
def generate_tts(input_text, language):
    model, tokenizer = tts_models[language]
    outputs = []
    for chunk in input_text.split("\n"):
        inputs = tokenizer(chunk, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).waveform
        outputs.append(output.float().numpy())
    combined_output = np.concatenate(outputs, axis=-1)
    sampling_rate = model.config.sampling_rate or 22050
    return combined_output, sampling_rate

# Main app function
def main():
    st.markdown("<h1 style='color: #00bfff;'>Audio/Video Transcription & Text-to-Speech</h1>", unsafe_allow_html=True)

    # Tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Transcription", "Text-to-Speech", "YouTube Video Transcription"])

    # Tab for audio transcription
    with tab1:
        st.subheader("Transcribe Audio File")
        uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])
        st.audio(uploaded_file)

        if uploaded_file:
            if st.button("Transcribe Audio"):
                with st.spinner("Processing..."):
                    start_time = time.time()

                    # Temporary file storage
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_file.write(uploaded_file.getbuffer())
                        temp_file_path = temp_file.name

                    # Convert to WAV using pydub
                    wav_file = convert_to_wav(temp_file_path)

                    # Perform transcription
                    transcription = transcription_pipe(wav_file)
                    formatted_transcription, full_transcription = format_transcription(transcription)

                    st.success("Transcription completed!")
                    st.subheader("Formatted Transcription with Timestamps")
                    st.text_area("Formatted Output", value=formatted_transcription, height=400)

                    st.subheader("Full Combined Transcription")
                    st.text_area("Combined Text Output", value=full_transcription, height=400)

                    # Download options
                    st.download_button("Download Formatted Transcription", formatted_transcription, file_name="formatted_transcription.txt")
                    st.download_button("Download Full Transcription", full_transcription, file_name="full_transcription.txt")

                    end_time = time.time()
                    st.write(f"Time taken: {round(end_time - start_time, 2)} seconds")

                    # Clean up
                    os.remove(temp_file_path)
                    os.remove(wav_file)

    # Tab for text-to-speech
    with tab2:
        st.subheader("Text-to-Speech")
        language = st.radio("Select Language", options=["English", "Hindi"])
        input_text = st.text_area("Enter text for conversion")

        if st.button("Convert to Speech"):
            if input_text:
                try:
                    output, rate = generate_tts(input_text, language)
                    st.audio(output, format="audio/wav", sample_rate=rate)
                except Exception as e:
                    st.error(f"Error generating speech: {e}")
            else:
                st.warning("Please enter text to convert.")

    # Tab for YouTube video transcription
    with tab3:
        st.subheader("Transcribe YouTube Video")
        video_url = st.text_input("Enter YouTube video link:")

        if video_url:
            if st.button("Transcribe Video"):
                with st.spinner("Processing..."):
                    try:
                        audio_file = download_audio_youtube(video_url)

                        if audio_file:
                            wav_file = convert_to_wav(audio_file)
                            transcription = transcription_pipe(wav_file)
                            formatted_transcription, full_transcription = format_transcription(transcription)

                            st.success("Transcription completed!")
                            st.subheader("Formatted Transcription with Timestamps")
                            st.text_area("Formatted Output", value=formatted_transcription, height=400)

                            st.subheader("Full Combined Transcription")
                            st.text_area("Combined Text Output", value=full_transcription, height=400)

                            st.download_button("Download Formatted Transcription", formatted_transcription, file_name="formatted_transcription.txt")
                            st.download_button("Download Full Transcription", full_transcription, file_name="full_transcription.txt")

                            os.remove(audio_file)
                            os.remove(wav_file)
                    except Exception as e:
                        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
