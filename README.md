---
title: Gradio Fastapi Static Server
emoji: 😻
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 43200
preload_from_hub:
  - m7n/intermediate_sci_pickle 100k_filtered_OA_sample_cluster_and_positions_supervised.pkl
  - m7n/intermediate_sci_pickle umap_mapper_250k_random_OA_discipline_tuned_specter_2_params.pkl
  - m7n/discipline-tuned_specter_2_024 model.safetensors
---




Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference