import os

from write_video import VideoProcessor
from cut_video import cut_video

file_path = 'input.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()

# Process each line in a loop
for line in lines:
    # Remove any leading/trailing whitespace
    line = line.strip()

    # Cut the videos
    video_file = line
    text_file = "couples.txt"
    processor = VideoProcessor(video_file, text_file)
    extraction = processor.process_video()
    
    #extract the data
    output_folder = "VATE"
    if extraction:
        cut_video(text_file, output_folder)

os.remove(text_file)
   
