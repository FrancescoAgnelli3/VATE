import os
from moviepy.editor import VideoFileClip
import speech_recognition as sr


def cut_video(text_file, output_folder):
    """
    Cuts a video into smaller segments based on the start and end times provided in a text file, and generates audio and text files for each segment.
    
    Args:
        text_file (str): The path to the text file containing the start and end times for each video segment.
        output_folder (str): The path to the output folder where the cut video, audio, and text files will be saved.
    
    Returns:
        None
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    video_path = None

    with open(text_file, 'r') as f:
        for i, line in enumerate(f):
            start, end, video_path = line.strip().split()
            start, end = float(start), float(end)
            print(f"Start: {start}, End: {end}")  # Debug print
            
            # Assuming the videos are named like video1.mp4, video2.mp4, etc.
            # video_file = f"video{i + 1}.mp4"
            output_path = os.path.join(output_folder, f"{video_path.split('/')[-1]}_{i+1}.mp4")
            output_audio_path = os.path.join(output_folder, f"{video_path.split('/')[-1]}_{i+1}.wav")
            output_text_path = os.path.join(output_folder, f"{video_path.split('/')[-1]}_{i+1}.txt")

            # Load the video using MoviePy
            video_clip = VideoFileClip(video_path)
            
            # Perform the video cut without re-encoding
            subclip = video_clip.subclip(start, end)
            
            # Save the cut video
            subclip.write_videofile(output_path, codec="libx264", audio_codec="aac")
            subclip.audio.write_audiofile(output_audio_path, codec="pcm_s16le")

            # Initialize recognizer class (for recognizing the speech)
            recognizer = sr.Recognizer()

            # Load audio file
            audio_file = sr.AudioFile(output_audio_path)

            # Convert audio to audio data
            with audio_file as source:
                audio_data = recognizer.record(source)

            # Recognize (convert from speech to text)
            try:
                text = recognizer.recognize_google(audio_data)
            except:
                text = ""
            with open(output_text_path, "w") as file:
                file.write(f"{text}\n")
        
            
            # Explicitly close the subclip
            subclip.close()

            # Close the video clip
            video_clip.close()
    
    if os.path.exists(video_path) and video_path is not None:
        os.remove(video_path)
        print(f'File {video_path} successfully deleted.')

    

    
