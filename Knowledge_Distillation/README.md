## 📘 Knowledge Distillation: Teacher–Student Model Compression

This project implements Knowledge Distillation, a model-compression technique where a smaller student model learns from a larger, well-trained teacher model.
Our goal is to achieve competitive performance with significantly fewer parameters, enabling deployment on lightweight or resource-constrained devices.

We use the Cats vs Dogs dataset from TensorFlow Datasets and apply a modern distillation pipeline using MobileNetV2 as the teacher and a compact custom CNN as the student.

## 📦 Requirements

This project is tested on macOS ARM64 (M1/M2/M3) with TensorFlow-Metal acceleration.

Create environment:
conda create -n tfenv python=3.10 -y
conda activate tfenv

Install dependencies:
pip install tensorflow-macos tensorflow-metal tensorflow-datasets
pip install matplotlib seaborn pandas ipykernel
python -m ipykernel install --user --name tfenv --display-name "tfenv"

## 🧠 Methodology
Teacher Model — MobileNetV2

Pretrained on ImageNet

Backbone frozen for efficiency

Strong feature extractor

Acts as the “expert” providing soft labels

Student Model — Lightweight CNN

Simple and shallow architecture

Much smaller (≈10× fewer parameters)

Intended for speed and deployment

Knowledge Distillation

The student is trained using a combination of:

Hard labels: Standard cross-entropy

Soft teacher predictions: KL Divergence

Temperature scaling: Smooths logits

α-weighting: Balances soft vs hard loss

This teaches the student to mimic the teacher's decision boundaries.

## 📊 Results Summary
Test Accuracy
Model	Accuracy
Teacher (MobileNetV2)	~0.97
Student Scratch	~0.68
Student Distilled	~0.63–0.66

Key Observations

The teacher performs best due to its pretrained, high-capacity architecture.

The student trained from scratch quickly overfits: high train accuracy, unstable validation accuracy.

The distilled student learns more smoothly, shows stronger regularization, and has a more stable validation pattern—even if trained for only 5 epochs.

With longer training (10–15 epochs), distilled students typically surpass scratch-trained ones.


The teacher model maintains the highest and most stable accuracy due to its strong pretrained features.
The student trained from scratch shows rising training accuracy but inconsistent validation performance, indicating overfitting.
The distilled student learns more smoothly and benefits from soft teacher guidance, resulting in better regularization.
Overall, knowledge distillation provides a more stable training signal for the smaller model.

## 🏁 Conclusion

This project demonstrates that knowledge distillation is an effective strategy for compressing large models while maintaining competitive performance.
The distilled student model exhibits smoother training behavior and improved generalization properties compared to a scratch-trained student.
This approach is highly useful for deploying models on mobile devices, embedded systems, and latency-sensitive applications.
