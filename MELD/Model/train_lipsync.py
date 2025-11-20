import os
import cv2
import librosa
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from utils import (
    LipSyncNet,
    AVAlignLoss,
    compute_sync_score,
    prepare_video_tensor,
    compute_audio_features,
    LipSyncDataset,  
    SyncNetInstance,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def av_align(
    video_dir,
    audio_dir,
    cache_dir,
    syncnet_checkpoint,
    fixed_num_frames=32,
    max_dirs=None,
    min_confidence=0.05,
    max_confidence=10.0,
):
    os.makedirs(cache_dir, exist_ok=True)

    syncnet = SyncNetInstance().to(device)
    syncnet.loadParameters(syncnet_checkpoint)
    syncnet.eval()

    sample_dirs = [
        d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))
    ]
    sample_dirs.sort()
    if max_dirs is not None:
        sample_dirs = sample_dirs[:max_dirs]

    data_paths = []

    for sample_dir in tqdm(sample_dirs, desc='Preprocessing samples'):
        sample_path = os.path.join(video_dir, sample_dir)
        audio_path = os.path.join(audio_dir, sample_dir + '.wav')
        if not os.path.exists(audio_path):
            continue

        face_videos = [f for f in os.listdir(sample_path) if f.endswith('.mp4')]
        if len(face_videos) < 2:
            continue

        try:
            audio, sr = librosa.load(audio_path, sr=None)
        except Exception:
            continue

        face_sync_scores = {}
        face_frames_dict = {}

        for face_video in face_videos:
            face_video_path = os.path.join(sample_path, face_video)
            cap = cv2.VideoCapture(face_video_path)
            if not cap.isOpened():
                continue

            face_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized_frame = cv2.resize(frame_rgb, (224, 224))
                face_frames.append(resized_frame)
            cap.release()

            if not face_frames:
                continue

            sync_score = compute_sync_score(syncnet, face_frames, audio, sr)
            if sync_score is None:
                continue

            if not (min_confidence <= sync_score <= max_confidence):
                continue

            face_sync_scores[face_video] = sync_score
            face_frames_dict[face_video] = face_frames

        if len(face_sync_scores) < 2:
            continue

        sorted_faces = sorted(
            face_sync_scores.items(), key=lambda x: x[1], reverse=True
        )
        speaker_face, _ = sorted_faces[0]
        false_face, _ = sorted_faces[-1]

        speaker_frames = face_frames_dict[speaker_face]
        pos_video_tensor = prepare_video_tensor(speaker_frames, fixed_num_frames)
        pos_audio_tensor = compute_audio_features(
            audio, sr, fixed_num_frames=fixed_num_frames
        )

        false_frames = face_frames_dict[false_face]
        neg_video_tensor = prepare_video_tensor(false_frames, fixed_num_frames)

        sample_data = {
            'video_pos': pos_video_tensor,
            'video_neg': neg_video_tensor,
            'audio': pos_audio_tensor,
        }
        cache_filename = f'{sample_dir}_av_alignment.pt'
        cache_path = os.path.join(cache_dir, cache_filename)
        torch.save(sample_data, cache_path)
        data_paths.append(cache_path)

    return data_paths


def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=20, scheduler=None
):
    best_val_loss = float('inf')
    best_model_state = None
    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for (video_pos, video_neg, audio) in tqdm(
            train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'
        ):
            video_pos = video_pos.to(device)
            video_neg = video_neg.to(device)
            audio = audio.to(device)

            optimizer.zero_grad()
            with autocast():
                audio_feat = model.encode_audio(audio)
                video_pos_feat = model.encode_video(video_pos)
                video_neg_feat = model.encode_video(video_neg)
                loss = criterion(video_pos_feat, video_neg_feat, audio_feat)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for (video_pos, video_neg, audio) in val_loader:
                video_pos = video_pos.to(device)
                video_neg = video_neg.to(device)
                audio = audio.to(device)

                audio_feat = model.encode_audio(audio)
                video_pos_feat = model.encode_video(video_pos)
                video_neg_feat = model.encode_video(video_neg)
                loss = criterion(video_pos_feat, video_neg_feat, audio_feat)
                val_epoch_loss += loss.item()

        avg_val_loss = val_epoch_loss / len(val_loader)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}] '
            f'Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}'
        )

        if scheduler is not None:
            scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def main():

    video_dir = 'path/to/train_video'
    audio_dir = 'path/to/train_audio'
    cache_dir = 'path/to/av_align'
    syncnet_checkpoint = 'path/to/syncnet_v2.model'
    output_model_path =  '/path/to/lipsync_model.pth'

    fixed_num_frames = 32
    max_dirs = None
    batch_size = 12
    num_epochs = 20
    lr = 1e-4
    weight_decay = 1e-4
    margin = 1.5
    alpha = 0.7
    lr_schedule_tmax = 100

    if os.path.exists(cache_dir) and any(
        f.endswith('.pt') for f in os.listdir(cache_dir)
    ):
        print('Using existing cached audio–visual alignment samples.')
        data_paths = [
            os.path.join(cache_dir, f)
            for f in os.listdir(cache_dir)
            if f.endswith('.pt')
        ]
    else:
        print('No cache found. Preprocessing audio–visual alignment samples...')
        data_paths = av_align(
            video_dir=video_dir,
            audio_dir=audio_dir,
            cache_dir=cache_dir,
            syncnet_checkpoint=syncnet_checkpoint,
            fixed_num_frames=fixed_num_frames,
            max_dirs=max_dirs,
        )

    if not data_paths:
        print('No alignment samples were created. Exiting.')
        return

    np.random.shuffle(data_paths)
    total_samples = len(data_paths)
    train_size = int(0.9 * total_samples)
    train_paths = data_paths[:train_size]
    val_paths = data_paths[train_size:]

    train_dataset = LipSyncDataset(train_paths, transform=None)
    val_dataset = LipSyncDataset(val_paths, transform=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = LipSyncNet(fixed_num_frames=fixed_num_frames).to(device)
    criterion = AVAlignLoss(margin=margin, alpha=alpha)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=lr_schedule_tmax
    )

    best_model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=num_epochs,
        scheduler=scheduler,
    )

    torch.save(best_model.state_dict(), output_model_path)
    print(f'Saved lipsync model to {output_model_path}')


if __name__ == '__main__':
    main()
