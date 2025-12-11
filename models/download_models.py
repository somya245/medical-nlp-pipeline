#!/usr/bin/env python3
"""
Model Downloader for Medical NLP Pipeline
Downloads and sets up all required NLP models and datasets
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import json
import spacy
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import torch

class ModelDownloader:
    def __init__(self, base_dir="models", cache_dir=None):
        self.base_dir = Path(base_dir)
        self.cache_dir = cache_dir or Path.home() / ".cache" / "medical_nlp"
        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Model configurations
        self.models = {
            "spacy": {
                "en_core_web_sm": {
                    "type": "spacy",
                    "download_cmd": "python -m spacy download en_core_web_sm"
                },
                "en_core_sci_md": {
                    "type": "spacy_sci",
                    "download_url": "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_md-0.5.3.tar.gz",
                    "pip_install": True
                }
            },
            "transformers": {
                "sentiment": {
                    "model_name": "bhadresh-savani/distilbert-base-uncased-emotion",
                    "task": "text-classification",
                    "local_name": "distilbert-emotion"
                },
                "summarization": {
                    "model_name": "facebook/bart-large-cnn",
                    "task": "summarization",
                    "local_name": "bart-cnn-summarizer"
                },
                "medical_ner": {
                    "model_name": "samrawal/bert-base-uncased_clinical-ner",
                    "task": "token-classification",
                    "local_name": "bert-clinical-ner"
                }
            },
            "nltk": [
                "punkt",
                "stopwords",
                "wordnet",
                "averaged_perceptron_tagger"
            ]
        }
        
    def download_spacy_models(self):
        """Download and install spaCy models"""
        print("\n" + "="*50)
        print("Downloading spaCy models...")
        print("="*50)
        
        for model_name, model_info in self.models["spacy"].items():
            model_path = self.base_dir / "spacy" / model_name
            model_path.mkdir(exist_ok=True, parents=True)
            
            print(f"\nChecking {model_name}...")
            
            try:
                # Try to load the model
                spacy.load(model_name)
                print(f"✓ {model_name} already installed")
            except OSError:
                print(f"✗ {model_name} not found. Downloading...")
                
                if model_info["type"] == "spacy":
                    # Standard spaCy model
                    try:
                        subprocess.run(
                            ["python", "-m", "spacy", "download", model_name],
                            check=True,
                            capture_output=True,
                            text=True
                        )
                        print(f"✓ Successfully downloaded {model_name}")
                    except subprocess.CalledProcessError as e:
                        print(f"✗ Failed to download {model_name}: {e}")
                        print("Trying alternative method...")
                        os.system(f"pip install https://github.com/explosion/spacy-models/releases/download/{model_name}-3.7.0/{model_name}-3.7.0.tar.gz")
                
                elif model_info["type"] == "spacy_sci":
                    # Scientific/Medical spaCy model
                    if model_info.get("pip_install", False):
                        try:
                            subprocess.run(
                                ["pip", "install", model_info["download_url"]],
                                check=True,
                                capture_output=True,
                                text=True
                            )
                            print(f"✓ Successfully downloaded {model_name}")
                        except subprocess.CalledProcessError as e:
                            print(f"✗ Failed to download {model_name}: {e}")
                            # Try alternative download method
                            self._download_sci_spacy_model(model_name, model_info["download_url"])
                    else:
                        self._download_sci_spacy_model(model_name, model_info["download_url"])
    
    def _download_sci_spacy_model(self, model_name, url):
        """Alternative method to download scientific spaCy models"""
        try:
            import requests
            import tarfile
            
            print(f"Downloading {model_name} from {url}...")
            
            # Download the tar.gz file
            response = requests.get(url, stream=True)
            tar_path = self.cache_dir / f"{model_name}.tar.gz"
            
            with open(tar_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Extract
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=self.cache_dir)
            
            # Install
            extracted_dir = self.cache_dir / model_name
            if extracted_dir.exists():
                subprocess.run(
                    ["pip", "install", str(extracted_dir)],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"✓ Successfully installed {model_name}")
            
            # Clean up
            tar_path.unlink(missing_ok=True)
            
        except Exception as e:
            print(f"✗ Error downloading {model_name}: {e}")
            print("You can manually install with:")
            print(f"pip install {url}")
    
    def download_transformers_models(self):
        """Download and cache Hugging Face models"""
        print("\n" + "="*50)
        print("Downloading Transformer models...")
        print("="*50)
        
        transformers_dir = self.base_dir / "transformers"
        transformers_dir.mkdir(exist_ok=True, parents=True)
        
        for model_type, model_info in self.models["transformers"].items():
            model_name = model_info["model_name"]
            local_name = model_info["local_name"]
            model_path = transformers_dir / local_name
            
            print(f"\nChecking {model_name} ({model_type})...")
            
            if model_path.exists():
                print(f"✓ {local_name} already downloaded")
                continue
            
            print(f"Downloading {model_name}...")
            
            try:
                # Download tokenizer and model
                if model_info["task"] == "text-classification":
                    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(self.cache_dir))
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_name, cache_dir=str(self.cache_dir)
                    )
                    
                elif model_info["task"] == "summarization":
                    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(self.cache_dir))
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name, cache_dir=str(self.cache_dir)
                    )
                
                elif model_info["task"] == "token-classification":
                    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(self.cache_dir))
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_name, cache_dir=str(self.cache_dir)
                    )
                
                # Save locally
                model_path.mkdir(exist_ok=True, parents=True)
                tokenizer.save_pretrained(str(model_path))
                model.save_pretrained(str(model_path))
                
                # Save model info
                model_config = {
                    "model_name": model_name,
                    "task": model_info["task"],
                    "local_name": local_name,
                    "download_date": str(Path(model_path).stat().st_ctime)
                }
                
                with open(model_path / "model_info.json", "w") as f:
                    json.dump(model_config, f, indent=2)
                
                print(f"✓ Successfully downloaded and saved {model_name}")
                print(f"  Saved to: {model_path}")
                
            except Exception as e:
                print(f"✗ Error downloading {model_name}: {e}")
                print(f"  You can try downloading manually:")
                print(f"  from transformers import AutoModel")
                print(f"  model = AutoModel.from_pretrained('{model_name}')")
    
    def download_nltk_data(self):
        """Download NLTK datasets"""
        print("\n" + "="*50)
        print("Downloading NLTK data...")
        print("="*50)
        
        nltk_dir = self.base_dir / "nltk_data"
        nltk.data.path.append(str(nltk_dir))
        
        for dataset in self.models["nltk"]:
            print(f"\nChecking NLTK {dataset}...")
            try:
                nltk.data.find(f'tokenizers/{dataset}')
                print(f"✓ NLTK {dataset} already downloaded")
            except LookupError:
                print(f"Downloading NLTK {dataset}...")
                try:
                    nltk.download(dataset, download_dir=str(nltk_dir), quiet=False)
                    print(f"✓ Successfully downloaded NLTK {dataset}")
                except Exception as e:
                    print(f"✗ Error downloading NLTK {dataset}: {e}")
    
    def download_additional_medical_models(self):
        """Download additional medical-specific models"""
        print("\n" + "="*50)
        print("Downloading Medical NLP models...")
        print("="*50)
        
        medical_dir = self.base_dir / "medical"
        medical_dir.mkdir(exist_ok=True, parents=True)
        
        # Additional medical models to consider
        medical_models = [
            {
                "name": "ClinicalBERT",
                "huggingface_name": "emilyalsentzer/Bio_ClinicalBERT",
                "description": "BERT model pretrained on PubMed and MIMIC-III clinical notes"
            },
            {
                "name": "BioBERT",
                "huggingface_name": "dmis-lab/biobert-base-cased-v1.1",
                "description": "BERT model pretrained on biomedical text"
            },
            {
                "name": "Clinical XLNet",
                "huggingface_name": "abhi1nandy2/Clinical-XLNet",
                "description": "XLNet model fine-tuned on clinical text"
            }
        ]
        
        for model_info in medical_models:
            model_path = medical_dir / model_info["name"].lower().replace("-", "_")
            
            print(f"\nChecking {model_info['name']}...")
            
            if model_path.exists():
                print(f"✓ {model_info['name']} already downloaded")
                continue
            
            print(f"Info: {model_info['description']}")
            response = input(f"Download {model_info['name']}? (y/N): ").strip().lower()
            
            if response == 'y':
                try:
                    print(f"Downloading {model_info['name']}...")
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_info["huggingface_name"],
                        cache_dir=str(self.cache_dir)
                    )
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_info["huggingface_name"],
                        cache_dir=str(self.cache_dir)
                    )
                    
                    # Save locally
                    model_path.mkdir(exist_ok=True, parents=True)
                    tokenizer.save_pretrained(str(model_path))
                    model.save_pretrained(str(model_path))
                    
                    print(f"✓ Successfully downloaded {model_info['name']}")
                    
                except Exception as e:
                    print(f"✗ Error downloading {model_info['name']}: {e}")
                    print("  This model is optional. You can skip it.")
            else:
                print(f"Skipping {model_info['name']}")
    
    def download_pubmed_word_vectors(self):
        """Download PubMed word vectors if needed"""
        print("\n" + "="*50)
        print("Checking for PubMed word vectors...")
        print("="*50)
        
        vectors_url = "https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin"
        vectors_path = self.base_dir / "word_vectors" / "biosentvec.bin"
        vectors_path.parent.mkdir(exist_ok=True, parents=True)
        
        if vectors_path.exists():
            print("✓ BioSentVec word vectors already downloaded")
            return
        
        print("\nBioSentVec: Word vectors trained on PubMed and MIMIC-III clinical notes")
        print(f"Size: ~4.5GB")
        response = input("Download BioSentVec word vectors? (y/N): ").strip().lower()
        
        if response == 'y':
            try:
                import requests
                from tqdm import tqdm
                
                print("Downloading BioSentVec word vectors...")
                print("This may take a while due to file size (4.5GB)...")
                
                response = requests.get(vectors_url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                
                with open(vectors_path, 'wb') as f, tqdm(
                    desc="Downloading",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for data in response.iter_content(chunk_size=8192):
                        size = f.write(data)
                        pbar.update(size)
                
                print(f"✓ Successfully downloaded word vectors to {vectors_path}")
                
            except Exception as e:
                print(f"✗ Error downloading word vectors: {e}")
                print("  You can manually download from:")
                print("  https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/")
        else:
            print("Skipping word vectors download")
    
    def create_model_config(self):
        """Create configuration file with model paths"""
        print("\n" + "="*50)
        print("Creating model configuration...")
        print("="*50)
        
        config = {
            "model_paths": {
                "spacy": {
                    "en_core_web_sm": str(self.base_dir / "spacy" / "en_core_web_sm"),
                    "en_core_sci_md": str(self.base_dir / "spacy" / "en_core_sci_md")
                },
                "transformers": {
                    "sentiment": str(self.base_dir / "transformers" / "distilbert-emotion"),
                    "summarization": str(self.base_dir / "transformers" / "bart-cnn-summarizer"),
                    "medical_ner": str(self.base_dir / "transformers" / "bert-clinical-ner")
                },
                "nltk_data": str(self.base_dir / "nltk_data"),
                "word_vectors": str(self.base_dir / "word_vectors" / "biosentvec.bin")
            },
            "cache_dir": str(self.cache_dir)
        }
        
        config_path = self.base_dir / "model_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Model configuration saved to: {config_path}")
        
        # Also create a Python module for easy access
        module_path = self.base_dir / "__init__.py"
        with open(module_path, "w") as f:
            f.write(f'''
"""
Model configuration module for Medical NLP Pipeline
Auto-generated by download_models.py
"""

import json
from pathlib import Path

BASE_DIR = Path(r"{self.base_dir}")
CACHE_DIR = Path(r"{self.cache_dir}")

# Load configuration
with open(BASE_DIR / "model_config.json", "r") as f:
    CONFIG = json.load(f)

# Convenience functions
def get_model_path(model_type, model_name=None):
    """Get path to a specific model"""
    if model_type == "spacy":
        return CONFIG["model_paths"]["spacy"][model_name]
    elif model_type == "transformers":
        return CONFIG["model_paths"]["transformers"][model_name]
    elif model_type == "nltk":
        return CONFIG["model_paths"]["nltk_data"]
    elif model_type == "word_vectors":
        return CONFIG["model_paths"]["word_vectors"]
    else:
        raise ValueError(f"Unknown model type: {{model_type}}")

def list_available_models():
    """List all available models"""
    models = {{}}
    for model_type, paths in CONFIG["model_paths"].items():
        if isinstance(paths, dict):
            models[model_type] = list(paths.keys())
        else:
            models[model_type] = paths
    return models
''')
        
        print(f"✓ Python module created at: {module_path}")
    
    def verify_installations(self):
        """Verify all installations"""
        print("\n" + "="*50)
        print("Verifying installations...")
        print("="*50)
        
        verifications = []
        
        # Verify spaCy
        try:
            nlp = spacy.load("en_core_web_sm")
            verifications.append(("spaCy en_core_web_sm", "✓"))
        except:
            verifications.append(("spaCy en_core_web_sm", "✗"))
        
        # Verify transformers
        try:
            from transformers import pipeline
            verifications.append(("Transformers", "✓"))
        except:
            verifications.append(("Transformers", "✗"))
        
        # Verify NLTK
        try:
            nltk.data.find('tokenizers/punkt')
            verifications.append(("NLTK data", "✓"))
        except:
            verifications.append(("NLTK data", "✗"))
        
        # Verify torch
        try:
            import torch
            verifications.append((f"PyTorch {torch.__version__}", "✓"))
        except:
            verifications.append(("PyTorch", "✗"))
        
        # Print verification results
        print("\nVerification Results:")
        print("-" * 30)
        for component, status in verifications:
            print(f"{component:30} {status}")
        
        # Check GPU availability
        if 'torch' in sys.modules:
            if torch.cuda.is_available():
                print(f"\n✓ GPU available: {torch.cuda.get_device_name(0)}")
                print(f"  CUDA version: {torch.version.cuda}")
            else:
                print("\n⚠ No GPU detected. Running on CPU.")
        
        all_good = all(status == "✓" for _, status in verifications)
        if all_good:
            print("\n✅ All required components installed successfully!")
        else:
            print("\n⚠ Some components failed installation. Check errors above.")
    
    def cleanup_cache(self):
        """Clean up download cache"""
        print("\n" + "="*50)
        print("Cleaning up cache...")
        print("="*50)
        
        cache_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
        cache_size_mb = cache_size / (1024 * 1024)
        
        response = input(f"Cache size: {cache_size_mb:.1f}MB. Clean up? (y/N): ").strip().lower()
        
        if response == 'y':
            try:
                import shutil
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(exist_ok=True)
                print("✓ Cache cleaned up")
            except Exception as e:
                print(f"✗ Error cleaning cache: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download models for Medical NLP Pipeline")
    parser.add_argument("--base-dir", default="models", help="Base directory for model storage")
    parser.add_argument("--cache-dir", help="Cache directory for downloads")
    parser.add_argument("--skip-medical", action="store_true", help="Skip optional medical models")
    parser.add_argument("--skip-vectors", action="store_true", help="Skip word vectors download")
    parser.add_argument("--cleanup", action="store_true", help="Clean up cache after download")
    
    args = parser.parse_args()
    
    print("="*60)
    print("MEDICAL NLP PIPELINE - MODEL DOWNLOADER")
    print("="*60)
    
    downloader = ModelDownloader(args.base_dir, args.cache_dir)
    
    try:
        # Step 1: Download spaCy models
        downloader.download_spacy_models()
        
        # Step 2: Download Transformers models
        downloader.download_transformers_models()
        
        # Step 3: Download NLTK data
        downloader.download_nltk_data()
        
        # Step 4: Optional medical models
        if not args.skip_medical:
            downloader.download_additional_medical_models()
        
        # Step 5: Optional word vectors
        if not args.skip_vectors:
            downloader.download_pubmed_word_vectors()
        
        # Step 6: Create configuration
        downloader.create_model_config()
        
        # Step 7: Verify installations
        downloader.verify_installations()
        
        # Step 8: Optional cleanup
        if args.cleanup:
            downloader.cleanup_cache()
        
        print("\n" + "="*60)
        print("DOWNLOAD COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("1. Run the main pipeline: python main.py")
        print("2. Check the models/ directory for downloaded models")
        print("3. Edit config.yaml if you need to change model paths")
        
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during download: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()