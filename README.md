# CycleGAN-Based Image Translation

This project implements an unpaired image-to-image translation model based on **CycleGAN** architecture. It enables style translation between two visual domains without requiring paired datasets. The model was built and trained from scratch using TensorFlow and Keras.

---

## Features

* **CycleGAN architecture** with ResNet-based Generators and PatchGAN Discriminators
* **Cycle-consistency loss** and **identity loss** for unpaired image translation
* Adversarial training with **Least Squares GAN (LSGAN)** loss
* **Instance Normalization** and residual learning for stable training
* Modular code with separate training and inference pipelines
* Example test script for image translation
---
## Getting Started

### 1. Clone this Repository

```bash
git clone https://github.com/aayushman950/CycleGAN-image-translation.git
cd CycleGAN-image-translation
```

### 2. Setup Environment

* Python: 3.10
* TensorFlow: 2.10+
* TensorFlow Addons: 0.20.0
* Keras: 2.10.0
* TensorFlow Estimator: 2.10.0
* NumPy: 1.23.5

You can use Conda:

```bash
conda create -n cyclegan python=3.10
conda activate cyclegan
pip install tensorflow==2.10.1 tensorflow-addons==0.20.0 keras==2.10.0 tensorflow-estimator==2.10.0 numpy==1.23.5
```

### 3. Dataset Preparation

Download or prepare an unpaired image dataset structured like this:

```
dataset/
├── trainA/    # images from domain A (e.g., Monet paintings)
└── trainB/    # images from domain B (e.g., landscape photos)
```

Example dataset used:

* [Monet2Photo]([https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/](https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/?fbclid=IwY2xjawLGVmhleHRuA2FlbQIxMABicmlkETFNWERlM1paTW52WlBwQ05oAR6IiOXJxgGUyQM_gajErh_MR7AJUzpFbWFwCzZZkl44ATUc-Is_B_ibPiI6gQ_aem_ml5IZucqiSFlgv6ENKaINA)) from the original CycleGAN paper

Update `DATA_DIR` in `train.py` to point to your dataset folder.

### 4. Training

```bash
python train.py
```

Model checkpoints will be saved in `checkpoints/` directory. You can change training settings like `EPOCHS` and `BATCH_SIZE` in `train.py`.

### 5. Inference / Testing

Run the example test script:

```bash
python test_monet.py
```

This will load the trained model and perform style translation on a sample image (Monet → photo or photo → Monet).

The translated image will be saved in `output_images/`.

---

## Results
![image](https://github.com/user-attachments/assets/48ad723b-1caf-4d04-839b-3a32bb405081)
![image](https://github.com/user-attachments/assets/ee5f65e7-6b74-4988-a765-7f705ed77e9d)


---

## References

* [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — Zhu et al., 2017
* TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)
