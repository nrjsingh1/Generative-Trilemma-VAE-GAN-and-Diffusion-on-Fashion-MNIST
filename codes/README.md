# CSIT 595 Assignment 1

The complete assignment implementation is self-contained in [`codes/assignment1_experiments.ipynb`](codes/assignment1_experiments.ipynb). The notebook is organized in assignment order and includes the common evaluator, VAE, GAN, diffusion model, training, evaluation, and visualization code directly in separate cells.

## Run

1. Install dependencies: `python -m pip install -r requirements.txt`
2. Open the notebook in Jupyter or Google Colab.
3. Run the smoke-test configuration first. Set `CONFIG["full_run"] = True` for the final Colab Pro run.
4. Run all cells in order and save the notebook after execution so the training logs, metrics, tables, and figures remain visible.

The notebook downloads Fashion-MNIST through `torchvision`, converts images to `1 x 32 x 32`, and normalizes them to `[-1, 1]`. It trains an evaluator CNN and unconditional VAE, DCGAN, and DDPM-style models, then computes cFID, predicted-label entropy, KL divergence to the real label distribution, sample speed, and the 5,000-image generation time.

The experiment is intentionally unconditional: labels are used only by the evaluator after generation and never by a generator or denoiser.

When run from this `codes/` directory, all generated artifacts are written to `codes/assignment1_outputs/`, including the trained model checkpoint `model_weights.pt`. The report at the package root reads the outputs from that directory.
