# Medical NLP Pipeline

> A compact, configurable pipeline and Streamlit demo for extracting clinical entities and running simple downstream analyses on medical text.

**Repository layout**

```
medical-nlp-pipeline/
├── app.py                      # Streamlit demo app
├── main.py                     # Script to run the pipeline from CLI
├── config.yaml                 # Pipeline configuration (models, thresholds, paths)
├── requirements.txt            # Core Python packages
├── requirements_streamlit.txt  # Additional packages for the Streamlit app
├── src/                        # Python source code (pipeline components, utils)
│   ├── ner_extractor.py
│   └── ...
├── models/                     # Pretrained/saved model artifacts (do NOT commit large binaries)
├── output/                     # Example outputs, logs, or export files
├── tests/                      # Unit tests
└── README.md
```

---

## Project overview

This repository contains a small **medical natural language processing (NLP)** pipeline intended to:

* perform clinical named-entity recognition (NER) and simple concept extraction from free-text clinical notes,
* provide a lightweight Streamlit-based demo (`app.py`) for interactive exploration,
* be configurable through `config.yaml` so you can swap models, adjust thresholds, and change input/output paths.

The design priorities are clarity, reproducibility, and easy local testing — not production readiness. Use this as a starting point for experiments, prototyping, or teaching.

---

## Features

* Modular pipeline components under `src/` (tokenization → NER → post-processing → export)
* Streamlit UI for quick demonstration of model outputs and simple visualizations
* Configuration-driven (via `config.yaml`) so model and I/O changes don't require code edits
* Example tests in `tests/` to validate key functions

---

## Getting started

> These instructions assume a machine with Python 3.8+ and Git installed.

### 1) Clone the repository

```bash
git clone https://github.com/somya245/medical-nlp-pipeline.git
cd medical-nlp-pipeline
```

### 2) Create and activate a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

If you only want to run the pipeline (no Streamlit UI):

```bash
pip install -r requirements.txt
```

If you want the Streamlit demo UI (recommended for exploration):

```bash
pip install -r requirements_streamlit.txt
```

> Tip: If you encounter line-ending warnings on Windows (LF → CRLF), follow the repository guidance or set `git config --global core.autocrlf true`.

---

## Configuration (`config.yaml`)

`config.yaml` contains runtime settings such as model file paths, thresholds, and input/output folders. Example keys you may find (or add):

```yaml
model:
  ner_model_path: "models/ner_model.pkl"
  tokenizer: "spacy_en_core_web_sm"
thresholds:
  ner_confidence: 0.5
io:
  input_path: "data/input.txt"
  output_dir: "output/"
```

Adjust these values before running the pipeline, or pass overrides via CLI (if implemented).

---

## Usage

### Run the pipeline from the command line

```bash
python main.py --config config.yaml --input data/sample_notes.txt --output output/results.json
```

`main.py` should load configuration, run the NER/extraction pipeline on the provided input, and write results to `output/`.

### Start the Streamlit demo

```bash
streamlit run app.py
```

Open the URL printed by Streamlit (usually `http://localhost:8501`) and try pasting clinical text to see model predictions and visualizations.

---

## Development

### Run tests

```bash
pytest -q
```

### Linting / formatting

Use `flake8` / `black` (if listed in `requirements.txt`) to keep code consistent.

```bash
black src/ tests/ app.py
flake8 src/
```

---

## Recommendations / Notes

* **Do not commit large model binaries.** Instead, store model artifacts in cloud storage (S3, GCS) or in `models/` but add them to `.gitignore` if they are large.
* **Handle PHI with care.** If you test this pipeline with real clinical text, ensure you follow local policies and de-identify data appropriately. This repository is for demonstration and research — not a HIPAA-certified product.
* **Line endings on Windows:** If you see warnings like `LF will be replaced by CRLF`, that is a Git setting (`core.autocrlf`). See the repo's guidance or run:

```powershell
git config --global core.autocrlf true
```

---

## Common troubleshooting

* *Streamlit fails to start or errors on import:* Check that `requirements_streamlit.txt` is installed and your virtual environment is active.
* *Model file not found:* Verify `config.yaml` paths and that trained artifacts exist in `models/`.
* *Package version conflicts:* Create a fresh virtual environment and reinstall dependencies.

---

## Contribution

Contributions are welcome. Typical ways to help:

1. Raise an issue describing the bug or feature request.
2. Fork the repo, create a feature branch, and submit a PR with tests.

Suggested branch naming: `feature/<short-description>` or `fix/<short-description>`.

---

## License

Specify your license here (e.g., MIT, Apache-2.0). If you are unsure, add an `UNLICENSED` or ask your project owner. Example (MIT):

```
MIT License
<year> <author/organization>
```

---

## Author / Contact

* Maintainer: Somya (GitHub: `@somya245`)
* For questions, open an issue on the repository.

---

