# Summarizer

A Transformer-based research paper summarization project built from scratch using PyTorch. This project trains a custom Transformer model on the Arxiv Summarization dataset to generate concise summaries of research papers.

The tokenization was done using pretrained tokenizer.

## Model Architecture

The project implements the core Transformer components manually, including:

* Token Embeddings
* Sinudsoisal Positional Encoding
* Multi-Head Self Attention
* Encoder Layers
* Decoder Layers
* Feed Forward Networks (SwiGLU)
* Layer Normalization (Both pre and post LN)
* Attention Masking

The model follows the sequence-to-sequence Transformer architecture commonly used in NLP tasks.

## Tech Stack

* Python
* PyTorch
* Matplotlib

## Dataset

This project uses the Arxiv Summarization Dataset from Hugging Face: [Arxiv Summarization Dataset](https://huggingface.co/datasets/ccdv/arxiv-summarization)


The dataset contains:

* Research paper articles
* Corresponding abstracts

The data is already formatted for summarization tasks, making it suitable for Transformer training.

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Download and Save dataset:

```bash
python get_full_dataset.py
```

Train model:

```bash
python model_train.py
```

Compute inference/summarization:

```bash
python generation_based_inference.py
```

## Example Workflow

1. Load the Arxiv dataset
2. Tokenize the preprocessed data
3. Train the Transformer model
4. Visualize training performance using Matplotlib
5. Generate summaries from unseen papers 

## Project Structure

```bash
Summarizer/
│
├── get_full_dataset.py/
├── tokenization.py/
├── embeddings.py/
├── attention_block.py/
├── normalize_layer.py/
├── feed_forward_network.py/
├── utils.py/
├── single_encoder_block.py/
├── full_encoder.py/
├── single_decoder_block.py/
├── full_decoder.py/
├── building_blocks.py/
├── train_and_validation.py/
├── visualize.py/
├── model_train.py
├── loss_based_inference.py
├── generation_based_inference.py/
├── requirements.txt
└── README.md
```

## Future Improvements

* Add data preprocessing pipeline 
* Add GPU optimization
* Experiment with larger datasets and different training and evaluation techniques
* Convert the current model into a multimodal architecture to better understand the paper


## Acknowledgements

* [PyTorch](https://pytorch.org)
* [Hugging Face Datasets](https://huggingface.co/datasets/ccdv/arxiv-summarization)
* [Attention Is All You Need Paper](https://arxiv.org/abs/1706.03762)
* [MatPlotLib](https://matplotlib.org/)
