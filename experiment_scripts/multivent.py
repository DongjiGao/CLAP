#!/usr/bin/env python3

# 2024 Dongji Gao

import argparse
import json
from pathlib import Path

import librosa
import torch
import numpy as np

import laion_clap
from laion_clap.training.data import (
    get_audio_features,
    int16_to_float32,
    float32_to_int16,
)


def get_args():
    parser = argparse.ArgumentParser(description="Multimodal retrieval")
    parser.add_argument(
        "--wav-scp",
        type=str,
        help="wav.scp file",
    )
    parser.add_argument(
        "--class-file",
        type=str,
        help="class file",
    )
    parser.add_argument(
        "--seg-duration",
        type=float,
        default=5.0,
        help="Duration of the audio segment for sound event classification",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=48000,
        help="Sampling rate of the audio",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="output directory",
    )
    return parser.parse_args()


def load_model(device):
    model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
    model.load_ckpt()

    return model


def main():
    args = get_args()
    wav_scp = Path(args.wav_scp)
    class_file = Path(args.class_file)
    output_dir = Path(args.output_dir)
    sampling_rate = args.sampling_rate
    duration = args.seg_duration

    assert sampling_rate == 48000

    device = torch.device("cuda:0")

    model = load_model(device)

    class_index_dict = {v: k for k, v in json.load(open(class_file)).items()}

    with open(wav_scp) as ws:
        with open(output_dir / "sound_event_classification.txt", "w") as sec:

            # get text embeddings for different classes
            all_texts = ["This is a sound of " + t for t in class_index_dict.values()]
            text_embed = model.get_text_embedding(all_texts)
            text_embed = torch.tensor(text_embed).to(device)

            # get audio embeddings for audio segments
            for line in ws:
                utt_id, wav_path = line.strip().split()

                audio_waveform, _ = librosa.load(wav_path, sr=sampling_rate)
                audio_waveform = audio_waveform.reshape(1, -1)
                audio_data = torch.from_numpy(
                    int16_to_float32(float32_to_int16(audio_waveform))
                ).float()

                frames_per_segment = int(duration * sampling_rate)
                num_segments = audio_data.shape[1] // frames_per_segment

                for i in range(num_segments):
                    start_time = i * duration

                    if i == num_segments - 1:
                        cur_audio_data = audio_data[:, i * frames_per_segment :]
                        end_time = audio_data.shape[1] / sampling_rate
                    else:
                        cur_audio_data = audio_data[
                            :, i * frames_per_segment : (i + 1) * frames_per_segment
                        ]
                        end_time = start_time + duration

                    cur_audio_embed = model.get_audio_embedding_from_data(
                        x=cur_audio_data, use_tensor=True
                    )

                    ranking = torch.argsort(
                        cur_audio_embed @ text_embed.t(),
                        descending=True,
                    )
                    pred_class = class_index_dict[ranking[0][0].item()]

                    start_time = format(
                        int(format(start_time, "0.3f").replace(".", "")), "08d"
                    )
                    end_time = format(
                        int(format(end_time, "0.3f").replace(".", "")), "08d"
                    )
                    sec.write(f"{utt_id}-{start_time}-{end_time} {pred_class}\n")


if __name__ == "__main__":
    main()
