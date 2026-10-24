# VoltStore Support Assistant

A local Streamlit chat app for a fictional electronics store, "VoltStore". The model is **Qwen2.5-1.5B-Instruct fine-tuned with QLoRA** to reply to customer support tickets in a fixed format (greeting, numbered steps, sign-off) using VoltStore's policies. It runs entirely locally, with no external API calls.


## Model

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Base model path | `models/qwen2.5-1.5b-instruct` |
| LoRA adapter path | `adapter/` |
| Loading | 4-bit base model + `PeftModel.from_pretrained(base, "adapter")` |

## Features

- Chat interface built with `st.chat_message` and `st.chat_input`
- Conversation memory across turns using `st.session_state`
- Token-by-token streaming with `TextIteratorStreamer`
- Loading spinner and clear error messages for load failures and out-of-memory errors
- Works fully offline after the base model download

## Run the app

1. Clone the repository and create a virtual environment:
   ```
   git clone https://github.com/sttr47/qwen-voltstore-support-assistant.git
   cd qwen-voltstore-support-assistant
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install PyTorch with CUDA ([pytorch.org](https://pytorch.org/get-started/locally/)):
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   ```
3. Install the remaining requirements:
   ```
   pip install -r requirements.txt
   ```
4. Download the base model (about 3 GB, one time):
   ```
   python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='models/qwen2.5-1.5b-instruct')"
   ```
5. Start the app (the adapter is loaded from `adapter/`):
   ```
   streamlit run app.py
   ```

On Windows, if a `UnicodeDecodeError: 'charmap'` appears, run `$env:PYTHONUTF8 = "1"` first.

## Reproduce the fine-tuning run

Training runs on a free Google Colab T4 GPU using `notebooks/Fine-Tuned Model.ipynb`.

**Data** (`data/`), chat format with `system`, `user` and `assistant` messages:

| File | Purpose | Rows |
|---|---|---|
| `voltstore_train.jsonl` | training | 162 |
| `voltstore_val_heldout_topics.jsonl` | validation (topics held out of training) | *fill in* |

**Steps**

1. Open the notebook in Google Colab and select **Runtime > Change runtime type > T4 GPU**.
2. Install the libraries:
   ```
   !pip uninstall -y torchao
   !pip install -q "torchao>=0.16.0"
   !pip install -q -U "bitsandbytes>=0.46.1" trl peft
   ```
3. Upload `voltstore_train.jsonl` and `voltstore_val_heldout_topics.jsonl` to the Colab session.
4. Run the cells in order:
   1. Load the base model in 4-bit and generate answers for the test prompts (before training)
   2. Train the adapter with TRL `SFTTrainer`
   3. Save the adapter with `trainer.save_model("adapter")`
   4. Generate answers for the same test prompts (after training)
5. Download the `adapter/` folder and place it in the project root.

**Hyperparameters**

| Setting | Value |
|---|---|
| Quantization | 4-bit NF4, double quantization, fp16 compute |
| LoRA | r=16, alpha=64, dropout=0.05 |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Epochs | 3 |
| Batch size | 2 per device, gradient accumulation 8 |
| Learning rate | 2e-4, cosine schedule, weight decay 0.03 |
| Max sequence length | 1024 |
| Precision | `fp16=False`, `bf16=False` (the T4 does not support bf16) |
| Gradient checkpointing | enabled |
| Best checkpoint | lowest validation loss |
| Seed | 42 |
| Libraries | transformers, TRL `SFTTrainer`, PEFT, bitsandbytes |

Training time: *fill in*. First and last training loss: *fill in*. Validation loss per epoch: *fill in*.


## Before and after fine-tuning

Same prompts and system prompt for both models.

Answers are in the colab file.

