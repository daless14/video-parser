import os
import subprocess
import whisper
import json
import yt_dlp
import shutil
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from transformers import pipeline

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def download_youtube_video(youtube_url, output_video="media/video.mp4"):
    """
    Download the full video (with both video and audio).
    """
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': output_video,
        'merge_output_format': 'mp4',
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return output_video

def convert_to_wav(input_file, output_file="media/audio.wav"):
    """
    Extract audio from the video file using ffmpeg and convert it to 16kHz mono WAV.
    """
    cmd = [
        "ffmpeg",
        "-i", input_file,
        "-vn",  # no video
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_file
    ]
    subprocess.run(cmd, check=True)
    return output_file

def extract_frames(video_file, output_folder="media/frames", interval=10):
    """
    Extract one frame every 'interval' seconds from the video.
    """
    os.makedirs(output_folder, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i", video_file,
        "-vf", f"fps=1/{interval}",
        f"{output_folder}/frame_%04d.jpg"
    ]
    subprocess.run(cmd, check=True)
    return output_folder

def get_video_context(frames_folder):
    """
    Generate a descriptive caption for each extracted frame using an image-to-text model.
    This provides richer context than simply listing filenames.
    """
    captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
    captions = []
    for i, frame in enumerate(frame_files):
        image_path = os.path.join(frames_folder, frame)
        try:
            result = captioner(Image.open(image_path), max_new_tokens=40)
            caption = result[0]['generated_text']
        except Exception as e:
            caption = "No caption available."
        captions.append(f"Frame {i+1} ({frame}): {caption}")
    return "\n".join(captions)

def transcribe_audio(audio_wav, model_name="small"):
    """
    Transcribe the audio using Whisper.
    """
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_wav)
    return result

def create_sections_from_transcript(transcript_text, video_context, youtube_url):
    prompt = f"""
    You are a chef giving a step-by-step tutorial on how to make food.
    You have two sources of information:
    1. An audio transcript of the tutorial.
    2. A summary of key video frames with descriptive captions.

    Use both sources to break the tutorial into logical sections and explain as if you were instructing someone. For each section, produce a JSON object with:
    - "section_title": A short title summarizing the section.
    - "description": A detailed paragraph describing the step.
    - "supplies_needed": A list of strings for the supplies/tools/ingredients mentioned in that section and make sure to include quantities of each item (if none, return an empty list).
    - "thumbnail_link": A reference to the video frame (from the video context) that best represents this section.
    - "video_link": A YouTube link with a query parameter "t" set to the starting time (in whole seconds) for that section, using the base URL {youtube_url}.

    Return ONLY valid JSON with a top-level key "sections" that contains an array of these section objects.

    Don't return this in markdown format, just return as a string.

    Audio Transcript:
    {transcript_text}

    Video Context:
    {video_context}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    content = response.choices[0].message.content
    return json.loads(content)

if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=8ZFEonQ5db8" 

    # Wipe media folder before processing
    if os.path.exists("media"):
        shutil.rmtree("media")

    print("Downloading video...")
    video_file = download_youtube_video(youtube_url)

    print("Extracting audio from video...")
    audio_file = convert_to_wav(video_file)

    print("Extracting video frames...")
    frames_folder = extract_frames(video_file, output_folder="media/frames", interval=5)
    video_context = get_video_context(frames_folder)
    print("Video context:")
    print(video_context)

    print("Transcribing audio with Whisper...")
    transcript = transcribe_audio(audio_file)
    full_transcript = transcript.get("text", "")
    print("Transcript:")
    print(full_transcript)

    print("Creating structured sections from transcript and video context...")
    try:
        sections = create_sections_from_transcript(full_transcript, video_context, youtube_url)
        print(json.dumps(sections, indent=2))

        with open("output.json", "w", encoding="utf-8") as f:
            json.dump(sections, f, ensure_ascii=False, indent=2)

    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e)
    except Exception as e:
        print("Error occurred:", e)
