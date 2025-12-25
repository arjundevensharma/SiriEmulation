# SiriEmulation
ML voice command classification model through intent classification and NER using TensorFlow and a trained BERT model.

Implementation:
1. Performs token-level classification to extract named entities from the given voice command using a BIO tagging scheme.
2. Uses a trained BERT model to extract features of the voice command including dropout, intent classifier, and slot classifier layers to predict the overall intent of the voice command based on extracted named entities.

Limitations:
1. The BERT model is trained primarily on English so other models like XLM or CamemBERT would have to be used to process commands in other languages.
2. Existing biases in the trained BERT model have not been audited.
3. The BERT model has many parameters which can be very computationally intensive to run. Simpler architectures (e.g. CNNs or LSTMs) might have a better speed/accuracy trade-off.
