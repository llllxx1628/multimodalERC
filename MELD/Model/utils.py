import os
import cv2
import time
import math
import glob
import subprocess
import numpy as np
import librosa
import python_speech_features
from scipy import signal
from scipy.io import wavfile
from shutil import rmtree
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import tempfile
import soundfile as sf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SyncNetOptions:
    def __init__(self, tmp_dir, reference, batch_size=32, vshift=10):
        self.tmp_dir = tmp_dir
        self.reference = reference
        self.batch_size = batch_size
        self.vshift = vshift


def compute_shifted_pairwise_distances(video_features, audio_features, vshift=10):
    window_size = vshift * 2 + 1
    padded_audio = torch.nn.functional.pad(audio_features, (0, 0, vshift, vshift))
    distances = []
    for i in range(len(video_features)):
        distances.append(
            torch.nn.functional.pairwise_distance(
                video_features[[i], :].repeat(window_size, 1),
                padded_audio[i : i + window_size, :],
            )
        )
    return distances


class SyncFeatureExtractor(nn.Module):
    def __init__(self, fc_dim=1024):
        super().__init__()

        self.audio_cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),
            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(256, 512, kernel_size=(5, 4), padding=(0, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.audio_fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, fc_dim),
        )

        self.lip_cnn = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=0),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(96, 256, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(256, 512, kernel_size=(1, 6, 6), padding=0),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )

        self.lip_fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, fc_dim),
        )

    def forward_audio(self, x):
        mid = self.audio_cnn(x)
        mid = mid.view(mid.size(0), -1)
        out = self.audio_fc(mid)
        return out

    def forward_lip(self, x):
        mid = self.lip_cnn(x)
        mid = mid.view(mid.size(0), -1)
        out = self.lip_fc(mid)
        return out

    def forward_lip_feature(self, x):
        mid = self.lip_cnn(x)
        out = mid.view(mid.size(0), -1)
        return out

    def forward_aud(self, x):
        return self.forward_audio(x)

    def forward_lipfeat(self, x):
        return self.forward_lip_feature(x)


class SyncNetInstance(nn.Module):
    def __init__(self, fc_dim=1024):
        super().__init__()
        self.backbone = SyncFeatureExtractor(fc_dim).to(device)

    def evaluate(self, opt, videofile):
        self.backbone.eval()

        tmp_ref_dir = os.path.join(opt.tmp_dir, opt.reference)
        if os.path.exists(tmp_ref_dir):
            rmtree(tmp_ref_dir)
        os.makedirs(tmp_ref_dir, exist_ok=True)

        img_pattern = os.path.join(tmp_ref_dir, "%06d.jpg")
        cmd_img = [
            "ffmpeg",
            "-y",
            "-i",
            videofile,
            "-threads",
            "1",
            "-f",
            "image2",
            img_pattern,
        ]
        subprocess.call(cmd_img, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        audio_path = os.path.join(tmp_ref_dir, "audio.wav")
        cmd_aud = [
            "ffmpeg",
            "-y",
            "-i",
            videofile,
            "-async",
            "1",
            "-ac",
            "1",
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            audio_path,
        ]
        subprocess.call(cmd_aud, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        images = []
        flist = glob.glob(os.path.join(tmp_ref_dir, "*.jpg"))
        flist.sort()
        for fname in flist:
            images.append(cv2.imread(fname))

        if len(images) == 0:
            return None, None, None

        im = np.stack(images, axis=3)
        im = np.expand_dims(im, axis=0)
        im = np.transpose(im, (0, 3, 4, 1, 2))
        im_tensor = torch.from_numpy(im.astype(float)).float()

        sample_rate, audio = wavfile.read(audio_path)
        mfcc = list(zip(*python_speech_features.mfcc(audio, sample_rate)))
        mfcc = np.stack([np.array(i) for i in mfcc])
        cc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)
        cc_tensor = torch.from_numpy(cc.astype(float)).float()

        min_length = min(len(images), math.floor(len(audio) / 640))
        lastframe = min_length - 5
        if lastframe <= 0:
            return None, None, None

        video_features = []
        audio_features = []

        for i in range(0, lastframe, opt.batch_size):
            im_batch = [
                im_tensor[:, :, vframe : vframe + 5, :, :]
                for vframe in range(i, min(lastframe, i + opt.batch_size))
            ]
            im_in = torch.cat(im_batch, 0).to(device)
            im_out = self.backbone.forward_lip(im_in)
            video_features.append(im_out.data.cpu())

            cc_batch = [
                cc_tensor[:, :, :, vframe * 4 : vframe * 4 + 20]
                for vframe in range(i, min(lastframe, i + opt.batch_size))
            ]
            cc_in = torch.cat(cc_batch, 0).to(device)
            cc_out = self.backbone.forward_audio(cc_in)
            audio_features.append(cc_out.data.cpu())

        video_features = torch.cat(video_features, 0)
        audio_features = torch.cat(audio_features, 0)

        distances = compute_shifted_pairwise_distances(
            video_features, audio_features, vshift=opt.vshift
        )
        mean_distance = torch.mean(torch.stack(distances, 1), 1)

        minval, minidx = torch.min(mean_distance, 0)
        offset = opt.vshift - minidx
        confidence = torch.median(mean_distance) - minval

        framewise_dist = np.stack([dist[minidx].numpy() for dist in distances])
        framewise_conf = torch.median(mean_distance).numpy() - framewise_dist
        framewise_conf_smoothed = signal.medfilt(framewise_conf, kernel_size=9)

        dists_npy = np.array([dist.numpy() for dist in distances])
        return offset.numpy(), confidence.numpy(), dists_npy

    def extract_feature(self, opt, videofile):
        self.backbone.eval()

        cap = cv2.VideoCapture(videofile)
        images = []
        while True:
            ret, image = cap.read()
            if not ret:
                break
            images.append(image)
        cap.release()

        if len(images) == 0:
            return None

        im = np.stack(images, axis=3)
        im = np.expand_dims(im, axis=0)
        im = np.transpose(im, (0, 3, 4, 1, 2))
        im_tensor = torch.from_numpy(im.astype(float)).float()

        lastframe = len(images) - 4
        if lastframe <= 0:
            return None

        video_features = []
        for i in range(0, lastframe, opt.batch_size):
            im_batch = [
                im_tensor[:, :, vframe : vframe + 5, :, :]
                for vframe in range(i, min(lastframe, i + opt.batch_size))
            ]
            im_in = torch.cat(im_batch, 0).to(device)
            im_out = self.backbone.forward_lip_feature(im_in)
            video_features.append(im_out.data.cpu())

        video_features = torch.cat(video_features, 0)
        return video_features

    def loadParameters(self, path):
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage)
        self_state = self.backbone.state_dict()
        for name, param in loaded_state.items():
            if name in self_state:
                self_state[name].copy_(param)


class LipSyncNet(nn.Module):
    def __init__(self, fixed_num_frames=32):
        super().__init__()
        self.fixed_num_frames = fixed_num_frames

        self.video_net = nn.Sequential(
            nn.Conv3d(
                3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 3, 3), stride=(2, 2, 2)),
            nn.Conv3d(
                64, 128, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)
            ),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(
                128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)
            ),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((4, 1, 1)),
            nn.Flatten(),
        )
        self.video_fc = nn.Sequential(
            nn.Linear(256 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
        )

        self.audio_net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.audio_fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def encode_video(self, video):
        x = self.video_net(video)
        x = self.video_fc(x)
        return F.normalize(x, p=2, dim=1)

    def encode_audio(self, audio):
        x = self.audio_net(audio)
        x = self.audio_fc(x)
        return F.normalize(x, p=2, dim=1)

    def forward(self, video, audio):
        video_feat = self.encode_video(video)
        audio_feat = self.encode_audio(audio)
        return video_feat, audio_feat


class AVAlignLoss(nn.Module):
    def __init__(self, margin=1.5, alpha=0.5):
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, video_pos_feat, video_neg_feat, audio_feat):
        d_pos = F.pairwise_distance(video_pos_feat, audio_feat, p=2)
        d_neg = F.pairwise_distance(video_neg_feat, audio_feat, p=2)
        target = torch.ones_like(d_pos)
        ranking_loss = self.ranking_loss(d_neg, d_pos, target)
        alignment_loss = torch.mean(d_pos)
        loss = self.alpha * ranking_loss + (1 - self.alpha) * alignment_loss
        return loss


def compute_sync_score(syncnet, face_frames, audio_clip, sr):
    with tempfile.TemporaryDirectory() as tmp_dir:
        opt = SyncNetOptions(tmp_dir=tmp_dir, reference="reference", batch_size=32, vshift=10)
        audio_path = os.path.join(tmp_dir, "audio.wav")
        sf.write(audio_path, audio_clip, sr)

        video_path = os.path.join(tmp_dir, "video.mp4")
        height, width, _ = face_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 25
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        for frame in face_frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()

        combined_video_path = os.path.join(tmp_dir, "combined.mp4")
        command = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            combined_video_path,
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not os.path.exists(combined_video_path):
            return None

        try:
            offset, confidence, dists_npy = syncnet.evaluate(opt, combined_video_path)
        except Exception:
            return None

    return confidence


def prepare_video_tensor(face_frames, fixed_num_frames=32):
    num_current_frames = len(face_frames)
    if num_current_frames < fixed_num_frames:
        pad_frames = [
            np.zeros_like(face_frames[0])
            for _ in range(fixed_num_frames - num_current_frames)
        ]
        face_frames = face_frames + pad_frames
    elif num_current_frames > fixed_num_frames:
        face_frames = face_frames[:fixed_num_frames]

    video_tensor = (
        torch.from_numpy(np.stack(face_frames, axis=0))
        .permute(3, 0, 1, 2)
        .float()
    )
    return video_tensor


def compute_audio_features(audio, sr, fixed_num_frames=32):
    video_fps = 25
    samples_per_frame = int(sr / video_fps)
    fixed_audio_length = fixed_num_frames * samples_per_frame

    if len(audio) < fixed_audio_length:
        pad_length = fixed_audio_length - len(audio)
        audio_clip = np.pad(audio, (0, pad_length), "constant", constant_values=0)
    else:
        audio_clip = audio[:fixed_audio_length]

    audio_mel = librosa.feature.melspectrogram(
        y=audio_clip, sr=16000, n_mels=64, hop_length=160, n_fft=400
    )
    audio_mel_db = librosa.power_to_db(audio_mel, ref=np.max)
    fixed_time_steps = 20

    if audio_mel_db.shape[1] < fixed_time_steps:
        pad_width = fixed_time_steps - audio_mel_db.shape[1]
        audio_mel_db = np.pad(
            audio_mel_db,
            ((0, 0), (0, pad_width)),
            mode="constant",
            constant_values=0,
        )
    else:
        audio_mel_db = audio_mel_db[:, :fixed_time_steps]

    audio_tensor = torch.from_numpy(audio_mel_db).unsqueeze(0).float()
    return audio_tensor


class LipSyncDataset(Dataset):
    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data = torch.load(self.data_paths[idx])
        video_pos = data["video_pos"]
        video_neg = data["video_neg"]
        audio = data["audio"]

        if self.transform:
            video_pos = self.transform(video_pos)
            video_neg = self.transform(video_neg)

        return video_pos, video_neg, audio
