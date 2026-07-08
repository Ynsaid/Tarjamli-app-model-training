# Tarjimli 🌍🔤

Tarjimli is an Arabic–English translation project that fine-tunes the **Helsinki-NLP/opus-mt-ar-en** sequence-to-sequence model on a custom bilingual dataset using Hugging Face Transformers. [code_file:1]

## Overview

The goal of this project is to build a practical machine translation engine between Arabic and English that you can plug into mobile, web, or backend applications. [code_file:1]
It loads a pretrained MarianMT model, prepares a CSV dataset of Arabic–English sentence pairs, and trains the model with the `Trainer` API for efficient experimentation and deployment. [code_file:1]

## Features

- Fine-tuning of a pretrained Arabic→English MarianMT model on custom data instead of using generic out-of-the-box translations. [code_file:1]
- Fully scripted training with Hugging Face `Trainer`, `TrainingArguments`, and `DataCollatorForSeq2Seq`. [code_file:1]
- Support for modern accelerators via `bf16`, `tf32`, and the fused optimizer `adamw_torch_fused`. [code_file:1]
- Clean saving of the final model and tokenizer to a dedicated directory, plus optional ZIP export for distribution. [code_file:1]
- Simple inference pipeline ready to be used as a translation service in your app. [code_file:1]

## Project Structure

```text
Tarjimli/
├── translate_train.ipynb          # Jupyter notebook for training and inference
├── final_ar_en_dataset.csv        # Custom dataset with columns: ar, en
├── opus-ar-en-custom/             # Training outputs and checkpoints
│   └── final-model/               # Saved final model (weights + tokenizer)
└── final-model.zip                # Compressed archive of the final model
```

## Requirements

The notebook uses the following Python packages: [code_file:1]

```bash
pip install transformers datasets sentencepiece accelerate evaluate sacrebleu -q
pip uninstall -y torch torchaudio torchvision
pip install torch torchaudio --no-cache-dir -q
pip install "transformers==4.44.2" -q
```

The exact Transformers version is pinned to `transformers==4.44.2` to avoid breaking changes in newer releases. [code_file:1]

## Dataset

Training is done on a CSV file loaded from:

```text
/home/jovyan/projects/translate/final_ar_en_dataset.csv
```

The file is expected to contain two text columns: [code_file:1]

| Column | Description |
|--------|-------------|
| `ar`   | Source sentence in Arabic |
| `en`   | Target sentence in English |

The dataset is split using `train_test_split(test_size=0.1)`, so 90% of the data is used for training and 10% for validation. [code_file:1]

## Training Setup

The notebook loads the base model and tokenizer as:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Helsinki-NLP/opus-mt-ar-en"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

Then it tokenizes both the Arabic source and English target sentences with a maximum length of 128 tokens and prepares them for sequence-to-sequence training. [code_file:1]

Key training arguments: [code_file:1]

| Setting          | Value                 |
|------------------|-----------------------|
| Base model       | `Helsinki-NLP/opus-mt-ar-en` |
| Max length       | `128`                 |
| Train batch size | `64`                  |
| Eval batch size  | `64`                  |
| Learning rate    | `3e-4`                |
| Epochs           | `2`                   |
| Save steps       | `2000`                |
| Eval steps       | `2000`                |
| Logging steps    | `200`                 |
| Precision        | `bf16=True`, `tf32=True` |
| Optimizer        | `adamw_torch_fused`   |

The training run reported `global_step=35498`, a runtime of about `3916.8` seconds, and a final training loss of `1.2212`. [code_file:1]

## Saving the Model

After training, the model and tokenizer are saved to the following path inside the project: [code_file:1]

```python
save_path = "/home/jovyan/projects/translate/opus-ar-en-custom/final-model"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print("saved to:", save_path)
```

To make distribution easier, the notebook also compresses the `final-model` directory into a ZIP archive using `shutil.make_archive`. [code_file:1]

## Inference (Translation)

The notebook demonstrates how to reload the saved model and run inference with the Transformers `pipeline` API: [code_file:1]

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_path = "/home/jovyan/projects/translate/opus-ar-en-custom/final-model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

translator = pipeline("translation", model=model, tokenizer=tokenizer)

texts = [
    "اريد تجربت صنع طائرة شراعية لكن الامر يبدو صعبا نوعا ما",
    "أنا أدرس الذكاء الاصطناعي",
    "هذا نموذج ترجمة عربي إلى إنجليزي"
]

for text in texts:
    out = translator(text, max_length=128)
    print("AR:", text)
    print("EN:", out[0]["translation_text"])
```

Sample outputs observed in the notebook include: [code_file:1]

| Arabic input                                                   | English output                                              |
|----------------------------------------------------------------|-------------------------------------------------------------|
| `اريد تجربت صنع طائرة شراعية لكن الامر يبدو صعبا نوعا ما` | `I want to try and make a drone, but it's kind of hard.`    |
| `أنا أدرس الذكاء الاصطناعي`                                   | `I'm studying artificial intelligence.`                     |
| `هذا نموذج ترجمة عربي إلى إنجليزي`                            | `This is an Arab translation model for English .`          |

Note: The pipeline issues a warning that a hardware accelerator (GPU) is available but the model is placed on CPU because no explicit `device` argument is passed. [code_file:1]

## Future Work

- Add automatic evaluation with BLEU using the `sacrebleu` package. [code_file:1]
- Extend the system to support English→Arabic translation as well.
- Integrate the model into a REST API, mobile app, or web interface branded as **Tarjimli**.
- Experiment with larger datasets and more training epochs to improve translation quality.

## License

This repository builds on the `Helsinki-NLP/opus-mt-ar-en` model hosted on Hugging Face. [code_file:1]
Please make sure to review and respect the upstream model license and the licenses of any datasets you train on.

## Author

Developed by **Younes Aid**, AI/ML engineer specialized in computer vision and applied deep learning.
