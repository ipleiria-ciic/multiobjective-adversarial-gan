# Balancing Image Quality and Attack Effectiveness in Multi-Objective Adversarial Image Generation

---

<p align="center">
    <img src="Assets/CIIC-FCT.png" width="75%"/>
</p>

---

## Description
Adversarial attacks present a serious challenge to deep neural networks (DNNs) in computer vision, introducing imperceptible perturbations that can mislead even state-of-the-art models. This project introduces a multi-objective generative adversarial network (GAN), augmented with an encoder, designed to generate adversarial images that balance attack effectiveness and visual quality.

Our approach was trained on data produced by four different adversarial attacks at varying perturbation levels and tested across five diverse DNN architectures. Evaluation metrics include not only fooling rate (FR) but also Fréchet Inception Distance (FID) and Learned Perceptual Image Patch Similarity (LPIPS) to assess image quality.

The model achieved a fooling rate of up to 89.63%, while maintaining high perceptual quality, with LPIPS as low as 0.23 and FID scores down to 25-demonstrating a strong trade-off between deception and image fidelity.

## Repository Structure
```
multiobjective-adversarial-gan/
│
├── 🎨 Assets/                  # Logos and visual assets
├── ⚔️ Attacks/                 # Implementations of adversarial attacks
├── 🧠 SuperstarGAN/            # Source code for SuperstarGAN and Encoder
├── 📓 Notebooks/               # Jupyter notebooks with pre-trained models
├── 🧪 Testing/                 # Scripts for testing and evaluation
├── 🙈 .gitignore               # Git ignore rules
├── 📦 requirements.txt         # Project dependencies
├── 🛠️ run_attacks.sh           # Script to generate perturbations via attacks
├── 🛠️ run_encoder.sh           # Script to train the encoder
├── 🛠️ run_superstargan.sh      # Script to train SuperstarGAN
├── 🛠️ run_testing.sh           # Script to test and evaluate generated images
├── 📜 README.md                # This documentation file
```

## Getting Started

To reproduce or extend this work, follow the steps below:

### 1. Generate Perturbations

Run:

```bash
run_attacks.sh
``` 

This script applies predefined adversarial attacks and generates the perturbed dataset. Alternatively, you may use your own attack code or pre-generated perturbations.

### 2. Train Adversarial GAN

Train the GAN using the perturbed dataset:

```bash
run_superstargan.sh
```
Want to tweak settings? You can modify the script to change the model, attack type, number of epochs, or delta values.

### 3. Train the Encoder

Once SuperstarGAN is trained, run:

```bash
run_encoder.sh
```

Ensure the script is pointing to the best GAN checkpoint.

### 4. Evaluate Performance

To test and evaluate the adversarial examples:

```bash
run_testing.sh
```

The output will include fooling rates and quality metrics, saved in a structured JSON format for further analysis.

## Metrics Used

- **Fooling Rate (FR)** – Measures the success rate of adversarial images in misleading target DNNs.
- **Fréchet Inception Distance (FID)** – Quantifies the visual quality of generated images.
- **Learned Perceptual Image Patch Similarity (LPIPS)** – Evaluates perceptual similarity between images.

## Acknowledgements
This work is funded by [Fundação para a Ciência e a Tecnologia](https://www.fct.pt/) through project UIDB/04524/2020.
