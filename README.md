# CD-GPT Fine-Tuning Repository

This code base contains a minimal, course-project-oriented subset of the official **CD-GPT** implementation.  It focuses on *parameter-efficient fine-tuning* (PEFT) of the pretrained CD-GPT backbone for transcription-factor binding prediction with Protein-Binding Microarray (PBM) data.

---
## 1  Repository Purpose
The goal of this repo is to demonstrate that a large biological foundation model can be adapted to small, specialised genomics data sets with very few trainable parameters, while still outperforming classical convolutional baselines such as DeepBind.

Concretely we:
1. load a frozen CD-GPT backbone;
2. inject a hybrid PEFT module combining **rsLoRA** (low-rank updates of QKV projections) and a **gated adapter** after each transformer block;
3. train only these lightweight layers on PBM measurements; and
4. evaluate single-TF and multi-TF regression performance.

---
## 2  Quick Start
### 2.1  Setup
```bash
conda create -n cdgpt_peft python=3.10
conda activate cdgpt_peft
pip install -r requirements.txt      # lightning, sklearn, sentencepiece, etc.
```

### 2.2  Prepare Data and Checkpoints
```
project/
├── data/                 # your TSV / CSV PBM files
├── checkpoints/
│   ├── CD-GPT-1b.pth     # pretrained backbone weights
│   └── tokenizer.model   # sentencepiece model
```

### 2.3  Single-GPU Training
```bash
python finetune_CDGPT.py \
    --data_dir     data/ \
    --checkpoint_dir checkpoints/ \
    --output_dir   outputs/ \
    --epochs       30 \
    --peft_rank    16
```
Key CLI flags:
* `--peft_rank`    LoRA rank (4–32 works in most cases; rank 8 used in paper).
* `--lr_lora`, `--lr_adapter`    separate LR for LoRA and adapter layers.
* `--batch_size 0` lets the script pick a size based on GPU RAM. 

After training, the best checkpoint (by validation R²) is saved under `outputs/` and can be evaluated with `pl.Trainer.test()`.

---
## 3  Role of the Modified Files
| File | Summary |
|------|---------|
| `finetune_CDGPT.py` | Entry-point Lightning script. Handles CLI, data loading, adaptive batch-size, training loop, and learning-rate schedule. |
| `model/finetune.py` | Defines `CDGPTFineTune`, which wraps the backbone, replaces self-attention with **LoRA** layers, adds **Adapter** modules, freezes original parameters, and exposes a sequence-level prediction head. |
| `DREAMdataset.py` | Custom `torch.utils.data.Dataset` for PBM TSV files. Normalises signal, constructs probe-specific tokens, and builds 4-D attention masks compatible with CD-GPT. |
| `evaluate_model.py` | Evaluate CD-GPT model performance and record prediction results. |
| `analysis/plot_metrics.py` | Utility to visualise training curves and predicted-vs-true scatter plots, producing the figure shown in the report. |

These components together form a minimal yet complete pipeline for PEFT on PBM data.

---
## 4  Citation
If you use this code for academic work, please cite the original CD-GPT preprint as well as the LoRA paper. 