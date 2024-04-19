# Model Explanation with LimeTabularExplainer

This project focuses on explaining a machine learning model that takes MIDI embeddings as input and predicts the genre of a musical piece. The model's predictions are interpreted using the **LimeTabularExplainer** from the [Lime](https://github.com/marcotcr/lime) library. This README explains the approach and the rationale for using this method.

## Project Overview

The project includes the following key components:

- **MIDI Embeddings**: The input data to the model consists of MIDI embeddings, which are numerical representations of musical data such as notes, tempo, time signature, and program. These embeddings are extracted from MIDI files.
  
- **Model**: The machine learning model (saved as a .h5 file) takes the MIDI embeddings as input and predicts the genre of the musical piece. The genres can range from classical to rock, pop, hip-hop, and more.

- **LimeTabularExplainer**: The model's predictions are interpreted using the LimeTabularExplainer from the Lime library.

## Explanation Method

### LimeTabularExplainer

LimeTabularExplainer was chosen to explain the model's predictions for the following reasons:

- **Feature Explanation**: LimeTabularExplainer provides insights into the contributions of individual features (embeddings) to the model's predictions. By perturbing the input data and observing the changes in model predictions, it generates explanations for the model's decision-making process.

- **Compatibility with MIDI Embeddings**: The MIDI embeddings are treated as features in a tabular dataset. Each dimension of the embeddings is considered a feature, making LimeTabularExplainer a suitable choice for explaining the model's predictions.

- **Local Explanations**: LimeTabularExplainer offers local explanations for specific instances, providing a detailed understanding of how the model arrives at a prediction for individual musical pieces.

- **Simplicity and Ease of Use**: The Lime library is user-friendly and provides straightforward methods for explaining model predictions, making it an accessible choice for this project.


