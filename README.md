# Tarjimli 🌍🔤

Tarjimli is an Arabic–English translation project that fine-tunes the **Helsinki-NLP/opus-mt-ar-en** sequence-to-sequence model on a custom bilingual dataset using Hugging Face Transformers. 

## Overview

The goal of this project is to build a practical machine translation engine between Arabic and English that you can plug into mobile, web, or backend applications. 
It loads a pretrained MarianMT model, prepares a CSV dataset of Arabic–English sentence pairs, and trains the model with the `Trainer` API for efficient experimentation and deployment. 

## Features

- Fine-tuning of a pretrained Arabic→English MarianMT model on custom data instead of using generic out-of-the-box translations. 
- Fully scripted training with Hugging Face `Trainer`, `TrainingArguments`, and `DataCollatorForSeq2Seq`. 
- Support for modern accelerators via `bf16`, `tf32`, and the fused optimizer `adamw_torch_fused`. 
- Clean saving of the final model and tokenizer to a dedicated directory, plus optional ZIP export for distribution. 
- Simple inference pipeline ready to be used as a translation service in your app. 

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

The notebook uses the following Python packages: 

```bash
pip install transformers datasets sentencepiece accelerate evaluate sacrebleu -q
pip uninstall -y torch torchaudio torchvision
pip install torch torchaudio --no-cache-dir -q
pip install "transformers==4.44.2" -q
```

The exact Transformers version is pinned to `transformers==4.44.2` to avoid breaking changes in newer releases. 

## Dataset

Training is done on a CSV file loaded from:

```text
https://www.kaggle.com/datasets/younesaid/translation-dataset-from-arabic-to-english
```

The file is expected to contain two text columns: 

| Column | Description |
|--------|-------------|
| `ar`   | Source sentence in Arabic |
| `en`   | Target sentence in English |

The dataset is split using `train_test_split(test_size=0.1)`, so 90% of the data is used for training and 10% for validation. 

## Training Setup

The notebook loads the base model and tokenizer as:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Helsinki-NLP/opus-mt-ar-en"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

Then it tokenizes both the Arabic source and English target sentences with a maximum length of 128 tokens and prepares them for sequence-to-sequence training. 

Key training arguments: 

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

The training run reported `global_step=35498`, a runtime of about `3916.8` seconds, and a final training loss of `1.2212`. 

## Saving the Model

After training, the model and tokenizer are saved to the following path inside the project: 

```python
save_path = "/home/jovyan/projects/translate/opus-ar-en-custom/final-model"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print("saved to:", save_path)
```

To make distribution easier, the notebook also compresses the `final-model` directory into a ZIP archive using `shutil.make_archive`. 

## Inference (Translation)

The notebook demonstrates how to reload the saved model and run inference with the Transformers `pipeline` API: 

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

Sample outputs observed in the notebook include: 

| Arabic input                                                   | English output                                              |
|----------------------------------------------------------------|-------------------------------------------------------------|
| `اريد تجربت صنع طائرة شراعية لكن الامر يبدو صعبا نوعا ما` | `I want to try and make a drone, but it's kind of hard.`    |
| `أنا أدرس الذكاء الاصطناعي`                                   | `I'm studying artificial intelligence.`                     |
| `هذا نموذج ترجمة عربي إلى إنجليزي`                            | `This is an Arab translation model for English .`          |

Note: The pipeline issues a warning that a hardware accelerator (GPU) is available but the model is placed on CPU because no explicit `device` argument is passed. 


## User Interface & Experience 

Tarjimli is designed with a clean, mobile-first interface focused on real-world translation workflows. The app provides a main text translation screen with source/target language selectors, a swap button, and quick actions for copying, listening to, and clear text.

![Splash screen](https://github.com/user-attachments/assets/59d241ec-394d-4d8c-b9be-a675f2b26eba)
![Translate screen](https://github.com/user-attachments/assets/526d2880-5a95-41f4-87c8-a2f32c81e460)

## Future Work

- Add support for multiple languages and enable bidirectional translation (Arabic ⇄ other languages).
- Extend the system to handle Arabic dialects and better represent diverse Arabic cultures.
- Improve the application’s ability to understand contextual meaning instead of relying on literal word‑for‑word translation.
- Enable real-time voice translation by integrating speech-to-text and text-to-speech technologies, rather than depending only on typed text input.

## License

This repository builds on the `Helsinki-NLP/opus-mt-ar-en` model hosted on Hugging Face. 
Please make sure to review and respect the upstream model license and the licenses of any datasets you train on.

## Author

Developed by **Younes Aid**, AI/ML engineer specialized in computer vision and applied deep learning.
