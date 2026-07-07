# Summarizer

A Transformer-based research paper summarization model built from scratch using PyTorch. The model is trained on the ArXiv Summarization dataset to generate concise summaries of research papers, using article body text as input and the abstract as the target.
```
Status: Training still in progress; model not fully converged yet.
```

## Model Architecture

All core Transformer components are implemented manually without relying on pre-built Transformer libraries.

```
              Component                           Value     

| d_model                                     |   512       |
| Encoder blocks                              |   5         |
| Decoder blocks                              |   10        |
| Attention heads (encoder / decoder / cross) |   8 / 8 / 8 |
| Feed-forward dimension (encoder & decoder)  |   1408      |
| Vocabulary size                             |   50,265    |
| Total parameters                            |   ~110M     |

```

**Components implemented:**

- Token embeddings with sinusoidal positional encoding
- Multi-head self-attention with separate Q, K, V projections and output projection
- SwiGLU feed-forward network — `FFN(x) = W3(SiLU(W1(x)) * W2(x))`
- Custom layer normalization with learnable scale and shift parameters
- Encoder stack (5 blocks) with padding mask
- Decoder stack (10 blocks) with causal mask, padding mask, and cross-attention
- Output projection from d_model to vocabulary size
- Gradient clipping (`max_norm=1.0`) and label smoothing (`0.1`) during training

The model follows a standard sequence-to-sequence Transformer architecture (encoder-decoder), similar to the original design from [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

## Tech Stack

- Python
- PyTorch
- Hugging Face Datasets & Transformers (tokenizer and dataset loading only)
- Matplotlib

## Dataset

This project uses the [ArXiv Summarization Dataset](https://huggingface.co/datasets/ccdv/arxiv-summarization) from Hugging Face.

- **Encoder input:** research paper article body
- **Decoder target:** paper abstract
- **Tokenizer:** `BartTokenizer` from `facebook/bart-base` (pre-trained, not fine-tuned)
- Data is pre-tokenized and saved locally under `./tokenized-dataset/{train, validation, test}`

**Note:** The dataset contains arXiv parsing artifacts (`@xmathXXX`, `@xcite`, HTML tags, encoded entities etc.) that were not cleaned prior to tokenization. A data cleaning pipeline is planned for a future training run.

## Training

Training uses a chunk-based approach — rather than iterating over the full dataset in a single epoch, the training data is processed in chunks of 1000 samples per call (`train_chunk_size=1000`). A true epoch completes when all chunks have been processed, after which the index resets to 0.

- **Optimizer:** AdamW (`lr=3e-4`, `weight_decay=1e-2`)
- **Scheduler:** ReduceLROnPlateau (`factor=0.8`, `patience=2`, `mode=min`)
- **Batch size:** 2 (with optional gradient accumulation)
- **Device:** MPS (Apple Silicon) 
- **Checkpointing:** saves model, optimizer, and scheduler state after every chunk, enabling seamless mid-cycle resumption

## Inference

Summarization is performed using beam search with three layers of output quality control:

- **Repetition penalty** — scales down the probability of previously generated tokens to discourage repetition
- **N-gram blocking** — blocks any token that would complete a previously seen n-gram sequence (configurable n)
- **Temperature scaling** — controls output diversity; lower temperature produces more focused summaries

Beam search experiments are parameterized through iterable hyperparameter lists, enabling exhaustive grid-search over combinations.

```
Note: Inference hyperparameters were tuned on a partially trained model and will be re-optimized after full convergence.

In addition, in the inference_variants file, I have added several different ways of decoding the transformer output for inference. As I explore more strategies, I will add more files there.
```
Each run outputs all completed beams ranked by cumulative log-probability, alongside the reference abstract for comparison.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

**Download and save dataset:**
```bash
python get_full_dataset.py
```

**Clean and save dataset:**
```bash
python clean_dataset.py
```

**Tokenize dataset:**
```bash
python tokenization.py
```

**Train model:**
```bash
python model_train.py
```

**Run beam search inference:**
```bash
python beam_search.py
```

## Example Workflow

1. Download the ArXiv dataset and save to disk
2. Clean the dataset, and save to disk
3. Tokenize the clan dataset articles and abstracts using the BART tokenizer
4. Train the encoder-decoder Transformer in chunks, saving checkpoints after each chunk
5. Monitor training and validation loss curves using Matplotlib
6. Generate summaries from unseen papers using beam search with repetition penalty and n-gram blocking

## Project Structure

```
Summarizer/
│
├── get_full_dataset.py               # download and save ArXiv dataset
├── clean_dataset.py                  # clean and save ArXiv dataset
├── tokenization.py                   # tokenize articles and abstracts
├── embeddings.py                     # token embeddings + sinusoidal positional encoding
├── attention_block.py                # multi-head attention
├── normalize_layer.py                # custom layer normalization
├── feed_forward_network.py           # SwiGLU feed-forward network
├── single_encoder_block.py           # single encoder block (attention + FFN + norm)
├── full_encoder.py                   # stacked encoder blocks
├── single_decoder_block.py           # single decoder block (masked att + cross att + FFN + norm)
├── full_decoder.py                   # stacked decoder blocks
├── building_blocks.py                # model initialization, checkpoint load/save, forward pass, train/val blocks
├── train_and_validation.py           # main training and validation loop
├── model_train.py                    # entry point for training runs
├── visualize.py                      # loss and perplexity curve plotting
├── inference_variants.py             # various inferences
    │
    ├── loss_based_inference.py       # teacher-forced loss-based inference
    ├── pure_greedy_decoding.py       # pure greedy decoding
    ├── beam_search.py                # decoding with beam search with n_gram and repetition penalty
├── utils.py                          # logging utilities, tensor debug helpers
├── requirements.txt
└── README.md
```

## Future Improvements

- **Data cleaning** — strip arXiv artifacts and retokenize the full dataset for a cleaner training run (in progress)
- **Reasoning traces** — fine-tune the model to generate structured `<think>...</think><summary>...</summary>` output, derived programmatically from existing abstracts (problem → contribution → method → results), with the reasoning span masked from the loss during training
- **Speculative decoding** — train a lightweight draft decoder (2–3 blocks, same d_model) to propose tokens ahead, with the full 10-block decoder verifying in a single parallel pass; particularly useful when combined with longer reasoning trace generation at inference time
- **Scheduler improvement** — replace fixed-interval `ReduceLROnPlateau` stepping with organic stepping tied to true validation loss stagnation
- **Multimodal extension** — incorporate figure and table understanding to better handle papers where key results are presented visually

## Acknowledgements

- [PyTorch](https://pytorch.org)
- [Hugging Face Datasets](https://huggingface.co/datasets/ccdv/arxiv-summarization)
- [Hugging Face Transformers](https://huggingface.co/transformers) (tokenizer)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Matplotlib](https://matplotlib.org)
