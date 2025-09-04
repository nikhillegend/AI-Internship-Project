# AI-Internship-Project
🖼️ Autoencoder Models for Image Reconstruction (CIFAR-10)

This project performs a comparative analysis of autoencoder architectures — Vanilla Autoencoder, Convolutional Autoencoder, and Variational Autoencoder (VAE) — on the CIFAR-10 dataset. The models were trained for image reconstruction tasks, with performance evaluated using reconstruction loss, MSE, inference time, and memory usage.

VAE achieved the best reconstruction quality (lowest loss).

Convolutional Autoencoder offered the most accurate pixel-level reconstruction (lowest MSE).

Vanilla Autoencoder proved most efficient, with fastest inference and minimal memory usage.
This project highlights trade-offs between accuracy, efficiency, and reconstruction quality in autoencoder design.
🔹 Tech Stack: Python, TensorFlow/Keras, NumPy, Matplotlib 

📝 Transformer Models for Question Answering (SQuAD)

A comparative NLP study on transformer architectures — BERT, RoBERTa, and DistilBERT — applied to the Stanford Question Answering Dataset (SQuAD). The project involved data preprocessing, model fine-tuning, and evaluation using Hugging Face’s Transformers library.

BERT delivered strong baseline accuracy.

RoBERTa outperformed others in robustness and contextual understanding.

DistilBERT offered faster inference with competitive accuracy, making it suitable for real-time use.
This project provides insights into selecting the best transformer model for QA tasks, balancing accuracy and computational efficiency.
🔹 Tech Stack: Python, Hugging Face Transformers, PyTorch, NLP 

📊 CNN vs Vision Transformers for Image Classification (CIFAR-10)

A comparative analysis of deep learning architectures on the CIFAR-10 dataset, evaluating 5 CNN models (AlexNet, VGG, ResNet, MobileNet) and Vision Transformer (ViT). Each model was trained with optimized hyperparameters and assessed on accuracy, training time, and resource usage.

ResNet & MobileNet offered the best trade-off between accuracy and efficiency.

VGG & AlexNet achieved high accuracy but required longer inference times.

Vision Transformer reached competitive accuracy but demanded more computational resources.
The study demonstrates strengths and weaknesses of CNNs vs Transformers in image classification, offering guidance for real-world deployment.
🔹 Tech Stack: Python, TensorFlow/Keras, PyTorch, NumPy, Matplotlib, Seaborn
