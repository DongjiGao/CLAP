#!/usr/bin/env bash

wav_scp_file="/exp/dgao/multivent_asr/data/emergency_data_en/wav.scp"
class_file="/exp/dgao/CLAP/class_labels/ESC50_class_labels_indices_space.json"
output_dir="/exp/dgao/multivent_asr/data/emergency_data_en/sound_event"

mkdir -p "${output_dir}"

./multivent.py \
    --wav-scp "${wav_scp_file}" \
    --class-file "${class_file}" \
    --output-dir "${output_dir}"

