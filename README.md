# RepoChat
Local Codebase Q&A with LLMs

## 🧰 Features

- Downloads GitHub repos and indexes their code
- Uses `llama.cpp` compatible models in `.gguf` format
- Automatically downloads models from HuggingFace using `models.json`
- Works fully offline after first setup

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## ▶️ Usage

```bash
python main.py --repo https://github.com/your/repo.git --model-key mistral
```

Available model keys are defined in `models.json`.

## 💡 Tip

The model will be automatically downloaded to `models/` using `huggingface_hub`.
