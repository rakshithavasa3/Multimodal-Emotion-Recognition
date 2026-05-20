from transformers import Wav2Vec2Model
from transformers import Wav2Vec2FeatureExtractor

model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

print("Model loaded successfully!")