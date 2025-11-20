import os
import json
import random
import cv2
import librosa
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    get_linear_schedule_with_warmup,
    Wav2Vec2Processor,
    Wav2Vec2Model,
    TimesformerModel,
    AutoImageProcessor,
)
from utils import (
    LipSyncNet,
    compute_audio_features,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DialogueEmotionModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super(DialogueEmotionModel, self).__init__()

        self.text_model = RobertaModel.from_pretrained(model_name)

        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.speaker_list = [
            "<s1>",
            "<s2>",
            "<s3>",
            "<s4>",
            "<s5>",
            "<s6>",
            "<s7>",
            "<s8>",
            "<s9>",
        ]
        self.speaker_tokens_dict = {"additional_special_tokens": self.speaker_list}
        self.tokenizer.add_special_tokens(self.speaker_tokens_dict)

        self.text_model.resize_token_embeddings(len(self.tokenizer))
        self.text_hidden_dim = self.text_model.config.hidden_size

        self.projection = nn.Linear(self.text_hidden_dim, 768)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        context_output = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, -1, :]
        hidden_state = self.projection(context_output)
        logits = self.classifier(hidden_state)
        return hidden_state, logits


def encode_right_truncated(text, tokenizer, max_length=511):
    tokenized = tokenizer.tokenize(text)
    truncated = tokenized[-max_length:]
    ids = tokenizer.convert_tokens_to_ids(truncated)
    return ids + [tokenizer.mask_token_id]


def pad_sequences(ids_list, tokenizer):
    max_len = max(len(ids) for ids in ids_list)

    padded_ids = []
    attention_masks = []

    for ids in ids_list:
        pad_len = max_len - len(ids)
        pad_ids = [tokenizer.pad_token_id for _ in range(pad_len)]
        attention_mask = [1 for _ in range(len(ids))]
        pad_attention = [0 for _ in range(len(pad_ids))]
        padded_ids.append(pad_ids + ids)
        attention_masks.append(pad_attention + attention_mask)

    return torch.tensor(padded_ids), torch.tensor(attention_masks)


def build_iemocap_context_input(dialogue_df, target_utterance_id):
    dialogue_df = dialogue_df.sort_values("Utterance_ID").reset_index(drop=True)

    target_row = dialogue_df[dialogue_df["Utterance_ID"] == target_utterance_id]
    if len(target_row) == 0:
        return None

    target_idx = target_row.index[0]

    input_string = ""
    current_speaker = None

    speaker_mapping = {"M": 1, "F": 2}
    unique_speakers = dialogue_df["Speaker"].unique()
    speaker_to_num = {s: i + 1 for i, s in enumerate(unique_speakers)}

    for idx in range(target_idx + 1):
        row = dialogue_df.iloc[idx]
        speaker = row["Speaker"]
        utt = str(row["Utterance"]).strip()

        if isinstance(speaker, str) and speaker in speaker_mapping:
            speaker_num = speaker_mapping[speaker]
        else:
            speaker_num = int(speaker_to_num.get(speaker, 1))

        input_string += "<s" + str(speaker_num) + "> "
        input_string += utt + " "
        current_speaker = speaker_num

    prompt = "Now" + " <s" + str(current_speaker) + "> " + "feels"
    concat_string = input_string.strip()
    concat_string += " " + "</s>" + " " + prompt

    return concat_string


class IEMOCAPDataset(Dataset):
    def __init__(self, df):
        self.emotion_list = [
            "neutral",
            "frustration",
            "sadness",
            "anger",
            "excited",
            "happiness",
        ]
        self.data = []

        for dialogue_id in df["Dialogue_ID"].unique():
            dialogue_df = df[df["Dialogue_ID"] == dialogue_id]

            for _, row in dialogue_df.iterrows():
                self.data.append(
                    {
                        "dialogue_df": dialogue_df,
                        "target_utterance_id": row["Utterance_ID"],
                        "emotion": row["Emotion"],
                    }
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def iemocap_collate_fn(batch, tokenizer):
    label_list = [
        "neutral",
        "frustration",
        "sadness",
        "anger",
        "excited",
        "happiness",
    ]

    batch_input = []
    batch_labels = []

    for sample in batch:
        dialogue_df = sample["dialogue_df"]
        target_utterance_id = sample["target_utterance_id"]
        emotion = sample["emotion"]

        concat_string = build_iemocap_context_input(dialogue_df, target_utterance_id)

        if concat_string is not None:
            encoded = encode_right_truncated(concat_string, tokenizer)
            batch_input.append(encoded)

            if emotion in label_list:
                label_index = label_list.index(emotion)
                batch_labels.append(label_index)
            else:
                continue

    if len(batch_input) == 0:
        return None

    batch_input_tokens, batch_attention_masks = pad_sequences(batch_input, tokenizer)
    batch_labels = torch.tensor(batch_labels)

    return batch_input_tokens, batch_attention_masks, batch_labels


def train_iemocap_text_encoder(
    input_csv, model_name, output_model_dir, batch_size, num_epochs, learning_rate
):
    df = pd.read_csv(input_csv, encoding="utf8")
    required_columns = {"Utterance", "Emotion", "Speaker", "Dialogue_ID", "Utterance_ID"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")

    standard_emotions = [
        "neutral",
        "frustration",
        "sadness",
        "anger",
        "excited",
        "happiness",
    ]
    df = df[df["Emotion"].isin(standard_emotions)].reset_index(drop=True)

    dialogue_ids = df["Dialogue_ID"].unique()
    train_dialogues, val_dialogues = train_test_split(
        dialogue_ids, test_size=0.1, random_state=42
    )

    train_df = df[df["Dialogue_ID"].isin(train_dialogues)].reset_index(drop=True)
    val_df = df[df["Dialogue_ID"].isin(val_dialogues)].reset_index(drop=True)

    model = DialogueEmotionModel(model_name, len(standard_emotions))
    model.to(device)

    train_dataset = IEMOCAPDataset(train_df)
    val_dataset = IEMOCAPDataset(val_df)

    def collate_wrapper(batch):
        return iemocap_collate_fn(batch, model.tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_wrapper,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_wrapper,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    num_training_steps = len(train_dataset) * num_epochs
    num_warmup_steps = len(train_dataset)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0

    print("Starting IEMOCAP text encoder training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            if batch_data is None:
                continue

            batch_input_tokens, batch_attention_masks, batch_labels = batch_data
            batch_input_tokens = batch_input_tokens.to(device)
            batch_attention_masks = batch_attention_masks.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            _, logits = model(batch_input_tokens, batch_attention_masks)

            loss = criterion(logits, batch_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        model.eval()
        val_preds, val_true = [], []

        with torch.no_grad():
            for batch_data in val_loader:
                if batch_data is None:
                    continue

                batch_input_tokens, batch_attention_masks, batch_labels = batch_data
                batch_input_tokens = batch_input_tokens.to(device)
                batch_attention_masks = batch_attention_masks.to(device)
                batch_labels = batch_labels.to(device)

                _, logits = model(batch_input_tokens, batch_attention_masks)

                pred_labels = logits.argmax(1).cpu().numpy()
                true_labels = batch_labels.cpu().numpy()

                val_preds.extend(pred_labels)
                val_true.extend(true_labels)

        if len(val_preds) > 0:
            val_f1 = f1_score(val_true, val_preds, average="weighted")
            avg_loss = total_loss / max(num_batches, 1)
            print(f"Epoch {epoch + 1}: Train Loss: {avg_loss:.4f}, Val F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                os.makedirs(output_model_dir, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "tokenizer": model.tokenizer,
                        "emotion_labels": standard_emotions,
                        "best_f1": best_val_f1,
                    },
                    os.path.join(output_model_dir, "iemocap_text_encoder.pth"),
                )
        else:
            print(f"Epoch {epoch + 1}: No valid validation samples")

    print(f"IEMOCAP training completed. Best Val F1: {best_val_f1:.4f}")
    return best_val_f1


def extract_text_features(
    input_csv, output_json, model_dir, batch_size=16, model_name="roberta-large"
):
    df = pd.read_csv(input_csv, encoding="utf8")
    required_columns = {"Utterance", "Dialogue_ID", "Utterance_ID", "Speaker"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")

    print(f"Processing {len(df)} utterances...")

    checkpoint_path = os.path.join(model_dir, "iemocap_text_encoder.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    emotion_labels = checkpoint["emotion_labels"]
    model = DialogueEmotionModel(model_name, len(emotion_labels))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = checkpoint["tokenizer"]

    features_dict = {}

    for dialogue_id in tqdm(df["Dialogue_ID"].unique(), desc="Processing dialogues"):
        dialogue_df = df[df["Dialogue_ID"] == dialogue_id]

        for _, row in dialogue_df.iterrows():
            target_utterance_id = row["Utterance_ID"]

            concat_string = build_iemocap_context_input(dialogue_df, target_utterance_id)

            if concat_string is not None:
                encoded = encode_right_truncated(concat_string, tokenizer)

                input_ids = torch.tensor([encoded], dtype=torch.long).to(device)
                attention_mask = torch.ones_like(input_ids).to(device)

                with torch.no_grad():
                    last_hidden, _ = model(input_ids, attention_mask)
                    feat = last_hidden.cpu().numpy()[0]

                key_name = f"dia{dialogue_id}_utt{target_utterance_id}"
                features_dict[key_name] = feat.tolist()

    with open(output_json, "w") as f:
        json.dump(features_dict, f, indent=4)

    print(f"IEMOCAP text features saved to {output_json}")


def extract_audio_features(
    input_dir,
    output_json,
    model_name="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    sample_rate=16000,
):
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.to(device)
    model.eval()

    audio_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".wav")]
    print(f"{input_dir} has {len(audio_files)} audio files")

    audio_features = {}
    for audio_file in tqdm(audio_files, desc="Processing audio"):
        path = os.path.join(input_dir, audio_file)
        base_name = os.path.splitext(audio_file)[0]
        try:
            wave, sr = librosa.load(path, sr=sample_rate)
        except Exception as exc:
            print(f"Error on {path}: {exc}")
            continue
        inputs = processor(
            wave, sampling_rate=sample_rate, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            hidden = out.last_hidden_state
            feat_vector = hidden.mean(dim=1).squeeze().cpu().numpy()
        audio_features[base_name] = feat_vector.tolist()

    with open(output_json, "w") as f:
        json.dump(audio_features, f)
    print(f"Audio features saved to {output_json}")


def extract_visual_features(
    video_dir,
    audio_dir,
    model_path,
    output_json="visual_features.json",
    fixed_num_frames=100,
    max_samples=None,
):
    model = LipSyncNet(fixed_num_frames=fixed_num_frames).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    timesformer_feature_extractor = AutoImageProcessor.from_pretrained(
        "facebook/timesformer-base-finetuned-k400"
    )
    timesformer_model = TimesformerModel.from_pretrained(
        "facebook/timesformer-base-finetuned-k400"
    )
    timesformer_model.eval()
    timesformer_model.to(device)

    augment = transforms.Compose(
        [
            transforms.ToPILImage(),
        ]
    )

    features_dict = {}
    sub_dirs = [
        d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))
    ]
    if max_samples is not None:
        sub_dirs = sub_dirs[:max_samples]

    for d in tqdm(sub_dirs, desc="Extracting visual features"):
        sample_path = os.path.join(video_dir, d)
        audio_file = os.path.join(audio_dir, d + ".wav")
        if not os.path.exists(audio_file):
            continue

        audio, sr = librosa.load(audio_file, sr=None)
        audio_tensor = compute_audio_features(
            audio, sr, fixed_num_frames=fixed_num_frames
        ).to(device)
        audio_tensor = audio_tensor.unsqueeze(0)

        face_videos = [f for f in os.listdir(sample_path) if f.endswith(".mp4")]
        if not face_videos:
            continue

        face_scores = {}
        frames_dict = {}
        for face_video in face_videos:
            video_path = os.path.join(sample_path, face_video)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = augment(frame_rgb)
                frames.append(frame_pil)
            cap.release()

            if not frames:
                continue

            count_diff = len(frames) - fixed_num_frames
            if count_diff < 0:
                extra = random.choices(frames, k=-count_diff)
                frames.extend(extra)
            elif count_diff > 0:
                frames = frames[:fixed_num_frames]

            pixel_values = timesformer_feature_extractor(
                frames, return_tensors="pt"
            )["pixel_values"].to(device)
            pixel_values_for_lipsync = pixel_values.permute(0, 2, 1, 3, 4)

            with torch.no_grad():
                video_feat, audio_feat = model(pixel_values_for_lipsync, audio_tensor)
                dist = nn.functional.pairwise_distance(video_feat, audio_feat)
                face_scores[face_video] = dist.item()
            frames_dict[face_video] = frames

        if face_scores:
            best_face = min(face_scores, key=face_scores.get)
            best_frames = frames_dict[best_face]

            pixel_values = timesformer_feature_extractor(
                best_frames, return_tensors="pt"
            )["pixel_values"].to(device)
            with torch.no_grad():
                outputs = timesformer_model(pixel_values)
                hidden_states = outputs.last_hidden_state
                vid_emb = hidden_states.mean(dim=1)
                vid_emb = vid_emb.squeeze(0).cpu().numpy()
            key_name = os.path.splitext(os.path.basename(audio_file))[0]
            features_dict[key_name] = vid_emb.tolist()

    with open(output_json, "w") as f:
        json.dump(features_dict, f)
    print(f"Visual features saved to {output_json}")


def main():
    datasets_info = [
        {
            "name": "train",
            "video_dir": "/path/to/iemocap/train_video",
            "audio_dir": "/path/to/iemocap/train_audio",
            "text_csv": "/IEMOCAP/Dataset/Data/train_iemo.csv",
            "output_feature_dir": "/IEMOCAP/Dataset/Data/train_features",
        },
        {
            "name": "val",
            "video_dir": "/path/to/iemocap/dev_video",
            "audio_dir": "/path/to/iemocap/dev_audio",
            "text_csv": "/IEMOCAP/Dataset/Data/dev_iemo.csv",
            "output_feature_dir": "/IEMOCAP/Dataset/Data/dev_features",
        },
        {
            "name": "test",
            "video_dir": "/path/to/iemocap/test_video",
            "audio_dir": "/path/to/iemocap/test_audio",
            "text_csv": "/IEMOCAP/Dataset/Data/test_iemo.csv",
            "output_feature_dir": "/IEMOCAP/Dataset/Data/test_features",
        },
    ]
    fine_tuned_model_dir = (
        "IEMOCAP/Dataset/Data/fine_tuned_model_iemo"
    )
    lipsync_model_path = "/path/to/lipsync_model.pth"

    train_dataset_info = next((d for d in datasets_info if d["name"] == "train"), None)

    if not os.path.exists(fine_tuned_model_dir):
        print("Starting IEMOCAP fine-tuning...")
        train_iemocap_text_encoder(
            input_csv=train_dataset_info["text_csv"],
            model_name="roberta-large",
            output_model_dir=fine_tuned_model_dir,
            batch_size=16,
            num_epochs=10,
            learning_rate=1e-5,
        )
    else:
        print(f"Found existing model in {fine_tuned_model_dir}")

    for d_info in datasets_info:
        output_dir = d_info["output_feature_dir"]
        os.makedirs(output_dir, exist_ok=True)

        text_features_json = os.path.join(output_dir, "text_features.json")

        extract_text_features(
            input_csv=d_info["text_csv"],
            output_json=text_features_json,
            model_dir=fine_tuned_model_dir,
            batch_size=16,
        )

        video_dir = d_info["video_dir"]
        audio_dir = d_info["audio_dir"]
        visual_features_json = os.path.join(output_dir, "visual_features.json")
        audio_features_json = os.path.join(output_dir, "audio_features.json")

        extract_visual_features(
            video_dir=video_dir,
            audio_dir=audio_dir,
            model_path=lipsync_model_path,
            output_json=visual_features_json,
        )

        extract_audio_features(
            input_dir=audio_dir,
            output_json=audio_features_json,
            model_name="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
            sample_rate=16000,
        )

    print("IEMOCAP processing completed!")


if __name__ == "__main__":
    main()
