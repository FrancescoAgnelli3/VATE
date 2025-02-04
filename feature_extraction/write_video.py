import cv2
import os
from moviepy.editor import VideoFileClip
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from pytubefix import YouTube
import argparse

class VideoProcessor:
    def __init__(self, video_file, text_file):
        self.video_file = video_file
        self.text_file = text_file
        
    def detect_faces(self, frame, face_detection):  
        """
        Detects faces in a given frame using the MediaPipe face detection model.
        
        Args:
            frame (numpy.ndarray): The input frame as a numpy array.
            face_detection (mediapipe.solutions.face_detection.FaceDetection): The MediaPipe face detection model.
        
        Returns:
            tuple: A tuple containing the bounding box coordinates and the RGB histograms of the detected face. If no face is detected, returns `None`.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        if results.detections and len(results.detections) == 1:
            bbox = results.detections[0].location_data.relative_bounding_box

            image_height, image_width, _ = frame.shape
            ymin, xmin, h, w = bbox.ymin, bbox.xmin, bbox.height, bbox.width
            # ymin, xmin, ymax, xmax = int(ymin * image_height), int(xmin * image_width), int((ymin + h) * image_height), int((xmin + w) * image_width)

            # 10% reduction of bounding box
            reduction_factor = 0.1
            new_h, new_w = int(h * (1 - reduction_factor)), int(w * (1 - reduction_factor))
            new_ymin, new_xmin = ymin + (h - new_h) // 2, xmin + (w - new_w) // 2
            new_ymin, new_xmin, new_ymax, new_xmax = int(new_ymin * image_height), int(new_xmin * image_width), int((new_ymin + new_h) * image_height), int((new_xmin + new_w) * image_width)

            # extraction of reducted bbox
            roi = rgb_frame[new_ymin:new_ymin + new_ymax, new_xmin:new_xmin + new_xmax]

            hist_r = cv2.calcHist([roi], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([roi], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([roi], [2], None, [256], [0, 256])

            # Normalizzare l'istogramma
            hist_r = cv2.normalize(hist_r, hist_r).flatten()
            hist_g = cv2.normalize(hist_g, hist_g).flatten()
            hist_b = cv2.normalize(hist_b, hist_b).flatten()

            curr_frame_bb, curr_frame_rgb = [new_xmin, new_ymin, new_xmax, new_ymax], [hist_r, hist_g, hist_b]

            return (curr_frame_bb, curr_frame_rgb)
        else:
            return None

    def clean_seconds(self, seconds_combined, fps):
        """
        Cleans the list of seconds by removing any seconds that are less than 1 second apart, and capping the maximum duration of each segment to 3 seconds.
        
        Args:
            seconds_combined (list): A list of seconds representing the start times of detected events.
            fps (float): The frames per second of the video.
        
        Returns:
            list: A list of tuples, where each tuple represents a cleaned segment of the video, containing the start and end times of the segment.
        """
        cleaned_seconds = []
        t_min,t_max = 6, 15
        for i in range(len(seconds_combined)-1):
            if (seconds_combined[i+1] - seconds_combined[i]) >= t_min:
                if (seconds_combined[i+1] - seconds_combined[i]) >= t_max:
                    cleaned_seconds.append((seconds_combined[i], seconds_combined[i] + t_max))
                    seconds_combined[i] = seconds_combined[i] + t_max
                else:
                    cleaned_seconds.append((seconds_combined[i], seconds_combined[i+1] - 1/fps))
            
        print("Secondi puliti:")
        print(cleaned_seconds)
        return cleaned_seconds

    def write_couples_txt(self, cleaned_seconds, video_path):
        """
        Writes the start and end times of cleaned video segments to a text file.
        
        Args:
            cleaned_seconds (list): A list of tuples, where each tuple represents a cleaned segment of the video, containing the start and end times of the segment.
            video_path (str): The file path of the video.
        
        Raises:
            IOError: If there is an error writing to the text file.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, self.text_file)
        try:
            with open(file_path, "w") as file:
                for i in range(len(cleaned_seconds)):
                    file.write(f"{cleaned_seconds[i][0]} {cleaned_seconds[i][1]} {video_path}\n")
        except IOError:
            print("Error: impossible to write on the file.")

    @staticmethod
    def calc_threshold(vector): 
        """
        Calculates a threshold value for a given vector.
        
        Args:
            vector (list): The input vector to calculate the threshold for.
        
        Returns:
            float: The calculated threshold value.
        
        This function first converts the input vector to a NumPy array and replaces any infinite values with 0. It then trims any leading or trailing zeros from the vector. The maximum value and mean value of the vector are calculated, and the threshold is set to be the average of these two values divided by 4.
        """
        vector = np.array(vector)
        vector[vector == np.inf] = 0
        vector = np.trim_zeros(vector)
        max_val = np.max(vector)
        mean_val = np.mean(vector)
        threshold = (max_val + mean_val) / 4
        return threshold
        
    @staticmethod
    def plotting(vectors, thresholds, names):
        """
        Plots the given vectors with their corresponding thresholds.
        
        Args:
            vectors (list): A list of vectors to be plotted.
            thresholds (list): A list of threshold values corresponding to the vectors.
            names (list): A list of names for the vectors.
        
        This function creates a set of subplots, one for each vector, and plots the vector along with a horizontal red line indicating the calculated threshold value. The title of each subplot is determined based on whether the vector is related to RGB or bounding box (BB) data.
        """
                # Creazione dei subplots
        fig, axs = plt.subplots(len(vectors), 1, figsize=(10, 8))

        for i, (vector, threshold, name) in enumerate(zip(vectors, thresholds, names)):

            # Creazione del grafico nel subplot corrente
            axs[i].plot(vector)

            # Aggiunta di una linea orizzontale rossa in corrispondenza del valore calcolato
            axs[i].axhline(y=threshold, color='red', linestyle='--')

            # Determinazione del titolo in base al nome del vettore
            if "RGB" in name:
                title = "RGB graphic"
            elif "BB" in name:
                title = "BB graphic"
            else:
                print("Error: no RGB or BB")
                return

            # Aggiunta di titolo e etichette per gli assi
            axs[i].set_xlabel('Index')
            axs[i].set_ylabel('Value')
            axs[i].set_title(title)

        # Ottimizzazione del layout
        plt.tight_layout()
        plt.show()

    def normalizing_vector(self):
        """
        Normalizes the RGB and bounding box (BB) vectors extracted from a video.
        
        This method downloads the video from a YouTube URL, extracts the RGB and BB vectors from the video frames, and normalizes them. It returns the normalized RGB and BB vectors, the video's FPS, and the path to the downloaded video file.
        
        Args:
            self (VideoProcessor): The instance of the VideoProcessor class.
        
        Returns:
            tuple: A tuple containing the normalized RGB vector, normalized BB vector, video FPS, and video file path.
        """
        try:
            yt = YouTube(self.video_file, use_oauth=True, allow_oauth_cache=True)
            title = yt.title
            title = title.replace(" ", "_")
            stream = yt.streams.get_highest_resolution()
            video_path = stream.download(filename = title)
            #documentation: https://pypi.org/project/pafy/

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            if not cap.isOpened():
                print("Error: impossible to open the video.")
                return
            
            currRGB, currBB = [], []
            prev_frame_bb, prev_frame_rgb = None, None
            norm_diff_rgb, norm_diff_bb = 0, 0

            mp_face_detection = mp.solutions.face_detection
            face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                current_frame = self.detect_faces(frame, face_detection)

                if current_frame is not None:
                    current_frame_bb, current_frame_rgb = current_frame

                    # frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    # if frame_index > 0:
                        # frame_index -= 1
                    # second = frame_index / fps

                    if current_frame_bb and prev_frame_bb and current_frame_rgb and prev_frame_rgb:
                        # Calcola la differenza nelle coordinate tra le due bbox
                        norm_diff_r = np.linalg.norm(current_frame_rgb[0] - prev_frame_rgb[0])
                        norm_diff_g = np.linalg.norm(current_frame_rgb[1] - prev_frame_rgb[1])
                        norm_diff_b = np.linalg.norm(current_frame_rgb[2] - prev_frame_rgb[2])

                        norm_diff_rgb = (norm_diff_r + norm_diff_g + norm_diff_b) / 3
                        currRGB.append(norm_diff_rgb)

                        # Calcola la differenza nelle coordinate tra le due bbox
                        diff_xmin = abs(current_frame_bb[0] - prev_frame_bb[0])
                        diff_ymin = abs(current_frame_bb[1] - prev_frame_bb[1])
                        diff_xmax = abs(current_frame_bb[2] - prev_frame_bb[2])
                        diff_ymax = abs(current_frame_bb[3] - prev_frame_bb[3])

                        # Calcola la norma della differenza
                        norm_diff_bb = np.linalg.norm([diff_xmin, diff_ymin, diff_xmax, diff_ymax])
                        currBB.append(norm_diff_bb)
                    else:
                        currBB.append(np.inf)
                        currRGB.append(np.inf)
                else:
                    current_frame_bb = None
                    current_frame_rgb = None
                    currBB.append(np.inf)
                    currRGB.append(np.inf)

                prev_frame_bb = current_frame_bb
                prev_frame_rgb = current_frame_rgb

            cap.release()
            cv2.destroyAllWindows()
        except:
            currBB = []
            currRGB = []
            fps = 0
            video_path = ""

        return currRGB, currBB, fps, video_path

    def process_video(self):
        """
        Processes a video by normalizing the RGB and bounding box vectors, calculating thresholds, and writing the extracted subvideos to a file.
        
        Args:
            None
        
        Returns:
            bool: True if subvideos were extracted, False otherwise.
        """
        vecRGB, vecBB, fps, video_path = self.normalizing_vector()
        if len(vecRGB)>0 and len(vecBB)>0:
            threshold_RGB = VideoProcessor.calc_threshold(vecRGB) 
            threshold_BB = VideoProcessor.calc_threshold(vecBB)
            print("threshold: ", threshold_BB, threshold_RGB)

            seconds_combined = []

            if len(vecRGB) == len(vecBB):
                for i in range(len(vecRGB)):

                    if vecRGB[i] > threshold_RGB or vecBB[i] > threshold_BB:
                        seconds_combined.append(i/fps)
            else:
                print("Error: vectors dimension not consistent")

            seconds_combined.insert(0, 0)
            seconds_combined = list(set(seconds_combined))
            seconds_combined.sort()

            cleaned_seconds = self.clean_seconds(seconds_combined, fps)
            self.write_couples_txt(cleaned_seconds, video_path)
            if len(cleaned_seconds) > 0:
                return True
            else:
                print("No subvideo extracted")
                os.remove(video_path)
                print(f'File {video_path} successfully deleted.')
                return False
        else:
            print("Video not extracted")
            return False

        #VideoProcessor.plotting([vecBB, vecRGB], [threshold_BB, threshold_RGB], ['currBB', 'currRGB'])

            
    def count_faces(self, frame, face_detection):  
        """
        Detects faces in a given frame using the MediaPipe face detection model.
        
        Args:
            frame (numpy.ndarray): The input frame as a numpy array.
            face_detection (mediapipe.solutions.face_detection.FaceDetection): The MediaPipe face detection model.
        
        Returns:
            tuple: A tuple containing the bounding box coordinates and the RGB histograms of the detected face. If no face is detected, returns `None`.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        if results.detections and len(results.detections) == 1:
            bbox = results.detections[0].location_data.relative_bounding_box
            # Calcola le coordinate del punto in alto a sinistra e in basso a destra
            image_height, image_width, _ = frame.shape
            ymin, xmin, h, w = bbox.ymin, bbox.xmin, bbox.height, bbox.width
            # ymin, xmin, ymax, xmax = int(ymin * image_height), int(xmin * image_width), int((ymin + h) * image_height), int((xmin + w) * image_width)

            # Riduzione della bounding box del 10%
            reduction_factor = 0.1
            new_h, new_w = int(h * (1 - reduction_factor)), int(w * (1 - reduction_factor))
            new_ymin, new_xmin = ymin + (h - new_h) // 2, xmin + (w - new_w) // 2
            new_ymin, new_xmin, new_ymax, new_xmax = int(new_ymin * image_height), int(new_xmin * image_width), int((new_ymin + new_h) * image_height), int((new_xmin + new_w) * image_width)

            # Estrarre la regione della bounding box ridotta
            roi = rgb_frame[new_ymin:new_ymin + new_ymax, new_xmin:new_xmin + new_xmax]

            # Calcolare l'istogramma dell'area delimitata
            hist_r = cv2.calcHist([roi], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([roi], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([roi], [2], None, [256], [0, 256])

            # Normalizzare l'istogramma
            hist_r = cv2.normalize(hist_r, hist_r).flatten()
            hist_g = cv2.normalize(hist_g, hist_g).flatten()
            hist_b = cv2.normalize(hist_b, hist_b).flatten()

            curr_frame_bb, curr_frame_rgb = [new_xmin, new_ymin, new_xmax, new_ymax], [hist_r, hist_g, hist_b]

            return (curr_frame_bb, curr_frame_rgb)
        else:
            return None
