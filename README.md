---
title: OpenAlex Mapper
emoji: 😻
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 5.23.1
python_version: "3.11"
app_file: app.py
pinned: true
# hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
# hf_oauth_expiration_minutes: 43200
preload_from_hub:
  - m7n/discipline-tuned_specter_2_024 100k_filtered_OA_sample_cluster_and_positions_supervised.pkl
  - m7n/discipline-tuned_specter_2_024 umap_mapper_250k_random_OA_discipline_tuned_specter_2_params.pkl
  - m7n/discipline-tuned_specter_2_024 model.safetensors
---
