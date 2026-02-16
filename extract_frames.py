import os
import cv2
from tqdm import tqdm

# dataset and output folders
DATASET_PATH = "dataset"
OUTPUT_PATH = "frames"

# number of frames to extract per video
MAX_FRAMES = 20


def extract_frames(video_path, save_folder):
    cap = cv2.VideoCapture(video_path)
    count = 0

    while cap.isOpened() and count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        frame_name = os.path.join(save_folder, f"frame_{count}.jpg")
        cv2.imwrite(frame_name, frame)
        count += 1

    cap.release()


for label in ["real", "fake"]:
    video_folder = os.path.join(DATASET_PATH, label)
    output_folder = os.path.join(OUTPUT_PATH, label)

    os.makedirs(output_folder, exist_ok=True)

    videos = os.listdir(video_folder)

    for video in tqdm(videos):
        video_path = os.path.join(video_folder, video)

        video_name = video.split(".")[0]
        save_folder = os.path.join(output_folder, video_name)
        os.makedirs(save_folder, exist_ok=True)

        extract_frames(video_path, save_folder)


print("✅ Frame extraction completed!")
