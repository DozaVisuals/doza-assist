# Third-Party Notices

This document contains attribution notices for open-source components and AI models
used by or distributed with Doza Assist.

---

## Gemma 4

**Copyright:** Google LLC

**License:** Apache License, Version 2.0
https://www.apache.org/licenses/LICENSE-2.0

**Model card:** https://huggingface.co/collections/google/gemma-4-release-6797d58b53b0d7fcba9a14ef

Gemma 4 model weights are **not bundled** in this repository or any release artifact.
They are downloaded at runtime by the user via Ollama (`ollama pull gemma4:<variant>`).
The specific variant is auto-selected based on the user's hardware; see `model_config.py`.

Use of Gemma 4 is subject to the Apache 2.0 license and any additional usage policies
published by Google at https://ai.google.dev/gemma/terms.

---

## OpenAI Whisper

**Copyright:** OpenAI

**License:** MIT License
https://github.com/openai/whisper/blob/main/LICENSE

Whisper is used as the non-English transcription fallback engine. It is installed as a
Python package dependency (`openai-whisper`) and is not bundled as a binary in this repo.

---

## Parakeet TDT 0.6B v2 (MLX Community conversion)

**Original model:** nvidia/parakeet-tdt-0.6b-v2
**Converted by:** MLX Community (https://huggingface.co/mlx-community)
**HuggingFace page:** https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2

**License:** Creative Commons Attribution 4.0 International (CC-BY-4.0)
https://creativecommons.org/licenses/by/4.0/

This model is used as the primary English transcription engine on Apple Silicon.
Model weights are downloaded at runtime via the `parakeet-mlx` Python package from
Hugging Face. They are not bundled in this repository.

Attribution: Original model by NVIDIA; MLX conversion by the MLX Community contributors.

---

## Ollama

**Copyright:** Ollama, Inc.

**License:** MIT License
https://github.com/ollama/ollama/blob/main/LICENSE

Ollama is a separate application that users install independently. This project does not
bundle or distribute Ollama. Users are responsible for complying with Ollama's license and
terms of service when using it to download and run AI models.

---

## Note to Users

Model weights for Gemma 4 and Parakeet are downloaded to your local machine at runtime.
By downloading these models you agree to their respective licenses. You are responsible
for ensuring your use complies with all applicable license terms and usage policies,
including any restrictions imposed by Google, NVIDIA, or Ollama.
