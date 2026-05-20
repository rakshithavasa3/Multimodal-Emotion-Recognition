# Multimodal Emotion Recognition

Recognizes emotions from Speech, Text, and both combined.

## Emotions Detected
angry, disgust, fear, happy, neutral, sad, surprise

## Dataset
TESS (Toronto Emotional Speech Set)

## Results
| Model | Accuracy |
|-------|----------|
| Speech Only (HuBERT) | 99.64% |
| Text Only (BERT) | 100.00% |
| Multimodal (Fusion) | 100.00% |

## How to Run

Install requirements:
pip install -r requirements.txt

Run Speech pipeline:
cd models/speech_pipeline
python train.py
python test.py

Run Text pipeline:
cd models/text_pipeline
python train.py
python test.py

Run Fusion pipeline:
cd models/fusion_pipeline
python train.py
python test.py

## Architecture
- Speech: HuBERT embeddings → MLP Classifier
- Text: BERT embeddings → MLP Classifier  
- Fusion: Concatenation of HuBERT + BERT → MLP Classifier