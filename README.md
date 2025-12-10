# ESE 3060 Final Project: NanoGPT Speedrun

**Team:** Evelyn Feng, Kate Hong  
**Course:** ESE 3060 

## Project Overview
This project investigates decoupled optimization strategies for the Language Model Head in the NanoGPT training pipeline. We hypothesized that the convex LM Head could be optimized more aggressively than the deep Transformer body. 

We replaced the standard AdamW optimizer for the head with **AdEMAMix** (Apple, 2024), a multi-scale momentum optimizer designed to separate "fast" exploration from "slow" exploitation. Our experiments benchmark this architecture on **8x A100 GPUs** to test for sample efficiency gains in speedrun.
## Repository Structure
* `train_ademamix.py`: The primary training script. Modified from the baseline `train_gpt.py` to support decoupled optimization.
* `ademamix.py`: A PyTorch implementation of the AdEMAMix optimizer.
* `train_gpt.py`: The original baseline script (for control experiments).
* `logs/`: Directory containing raw training logs for the 4-GPU and 8-GPU runs.

## Setup & Installation

1. **Environment:** This code is designed to run in the standard `modded-nanogpt` environment provided by the course (PyTorch 2.1+, CUDA 12.x).
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
3. **Running Files:** Download the FineWeb-Edu dataset (10 shards / 1.0B tokens):
   ```bash
   python cached_fineweb10B.py 10
   ```bash
   --standalone --nproc_per_node=8 train_ademamix.py

## Conclusion
We found that AdEMAMix introduces significant momentum drag in short-horizon training. The overhead of its slow momentum buffer prevents the model from making the rapid, sharp adjustments required in the final phase of the speedrun, causing it to trail the baseline slightly

## Acknowledgments
* Base code derived from Keller Jordan's modded-nanogpt
* AdEMAMix algorithm based on The AdEMAMix Optimizer: Better, Faster, Older (Pagliardini et al., 2024).
