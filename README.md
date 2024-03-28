# SiriEmulation
Comprehensive machine learning voice command classification model through intent classification and Named Entity Recognition (NER) using TensorFlow and a pre-trained Bidirectional Encoder Representations from Transformers (BERT) model. Developed in December 2022 alongside Inspirit AI.

## Details
The model achieves its function by:
1. Performing token-level classification to extract named entities from the given voice command using a BIO tagging scheme.
2. Using a pre-trained BERT model to extract features of the voice command including dropout, intent classifier, and slot classifier layers to predict the overall intent of the voice command based on extracted named entities.

## Limitations
1. The BERT model is pre-trained primarily on English so other models like XLM or CamemBERT would have to be used to process commands in other languages.
2. Existing biases in the pre-trained BERT model have not been audited.
3. The BERT model has many parameters which are very computationally intensive to run. Simpler architectures (e.g. CNNs or LSTMs) might have a better speed/accuracy trade-off.
