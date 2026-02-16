import os
import cv2
from mtcnn import MTCNN
from tqdm import tqdm

FRAMES_PATH = "frames"
OUTPUT_PATH = "faces"

detector = MTCNN()


def extract_face(image_path, save_path):
    img = cv2.imread(image_path)
    if img is None:
        return

    results = detector.detect_faces(img)

    if len(results) == 0:
        return

    x, y, w, h = results[0]['box']
    x, y = abs(x), abs(y)

    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (224, 224))

    cv2.imwrite(save_path, face)


for label in ["real", "fake"]:
    label_path = os.path.join(FRAMES_PATH, label)
    output_label_path = os.path.join(OUTPUT_PATH, label)

    os.makedirs(output_label_path, exist_ok=True)

    videos = os.listdir(label_path)

    for video in tqdm(videos):
        video_path = os.path.join(label_path, video)
        save_video_path = os.path.join(output_label_path, video)
        os.makedirs(save_video_path, exist_ok=True)

        images = os.listdir(video_path)

        for img_name in images:
            img_path = os.path.join(video_path, img_name)
            save_path = os.path.join(save_video_path, img_name)

            extract_face(img_path, save_path)


print("✅ Face extraction completed!")
