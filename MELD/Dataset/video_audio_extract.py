import os
import cv2
import fnmatch
import numpy as np
import subprocess
from mtcnn import MTCNN

detector = MTCNN(min_face_size=20, steps_threshold=[0.7, 0.7, 0.8])

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,
        0.03,
    ),
)


def convert_mp4_to_wav(mp4_path, output_wav_path):
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            mp4_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-y",
            output_wav_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def process_video_faces(video_path, output_video_base_dir):
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    output_video_dir = os.path.join(output_video_base_dir, video_filename)
    os.makedirs(output_video_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    face_id_counter = 0
    face_trackers = {}
    face_videos = {}

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    prev_gray = None
    prev_points = np.array([], dtype=np.float32).reshape(-1, 1, 2)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if prev_gray is None:
            faces = detector.detect_faces(rgb_frame)
            for face in faces:
                if face.get("confidence", 0.0) < 0.9:
                    continue

                x, y, w, h = face["box"]
                x1, y1, x2, y2 = x, y, x + w, y + h
                face_id_counter += 1

                tracker = cv2.TrackerKCF_create()
                bbox = (x1, y1, x2 - x1, y2 - y1)
                tracker.init(frame, bbox)
                face_trackers[face_id_counter] = (tracker, bbox)

                video_output_path = os.path.join(
                    output_video_dir,
                    f"face_{face_id_counter}.mp4",
                )
                face_videos[face_id_counter] = cv2.VideoWriter(
                    video_output_path,
                    fourcc,
                    fps,
                    (160, 160),
                )

                prev_points = np.append(
                    prev_points,
                    np.array([[x + w / 2, y + h / 2]], dtype=np.float32).reshape(
                        -1, 1, 2
                    ),
                    axis=0,
                )

            prev_gray = gray_frame
        else:
            if len(prev_points) > 0:
                next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray_frame, prev_points, None, **lk_params
                )
                if status is not None:
                    status = status.reshape(-1)
                    valid_points = status == 1
                    prev_points = next_points[valid_points].reshape(-1, 1, 2)
                else:
                    prev_points = np.array([], dtype=np.float32).reshape(-1, 1, 2)
            else:
                prev_points = np.array([], dtype=np.float32).reshape(-1, 1, 2)

            prev_gray = gray_frame

            new_trackers = {}
            for tracker_id, (tracker, bbox) in face_trackers.items():
                ok, bbox = tracker.update(frame)
                if ok:
                    x1, y1, w, h = map(int, bbox)
                    x2, y2 = x1 + w, y1 + h
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame_width, x2)
                    y2 = min(frame_height, y2)

                    face_frame = frame[y1:y2, x1:x2]
                    if face_frame.size != 0:
                        face_frame_resized = cv2.resize(face_frame, (160, 160))
                        face_videos[tracker_id].write(face_frame_resized)
                    new_trackers[tracker_id] = (tracker, bbox)
                else:
                    black_frame = np.zeros((160, 160, 3), dtype=np.uint8)
                    face_videos[tracker_id].write(black_frame)

            face_trackers = new_trackers

    cap.release()
    for video_writer in face_videos.values():
        video_writer.release()
    cv2.destroyAllWindows()


def process_meld_split(input_video_dir, output_video_dir, output_audio_dir):
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_audio_dir, exist_ok=True)

    video_files = [
        f
        for f in os.listdir(input_video_dir)
        if fnmatch.fnmatch(f, "dia*_utt*.mp4")
    ]

    for video_file in video_files:
        video_path = os.path.join(input_video_dir, video_file)

        process_video_faces(video_path, output_video_dir)

        video_base_name = os.path.splitext(video_file)[0]
        output_wav_path = os.path.join(output_audio_dir, f"{video_base_name}.wav")
        convert_mp4_to_wav(video_path, output_wav_path)


def main():
    datasets = [
        {
            "name": "train",
            "input_video_dir": "/path/to/MELD.Raw/output_repeated_splits_train",
            "output_video_dir": "/path/to/meld/train_video",
            "output_audio_dir": "/path/to/meld/train_audio",
        },
        {
            "name": "dev",
            "input_video_dir": "/path/to/MELD.Raw/dev",
            "output_video_dir": "/path/to/meld/dev_video",
            "output_audio_dir": "/path/to/meld/dev_audio",
        },
        {
            "name": "test",
            "input_video_dir": "/path/to/MELD.Raw/test",
            "output_video_dir": "/path/to/meld/test_video",
            "output_audio_dir": "/path/to/meld/test_audio",
        },
    ]

    for cfg in datasets:
        if os.path.isdir(cfg["input_video_dir"]):
            process_meld_split(
                input_video_dir=cfg["input_video_dir"],
                output_video_dir=cfg["output_video_dir"],
                output_audio_dir=cfg["output_audio_dir"],
            )


if __name__ == "__main__":
    main()