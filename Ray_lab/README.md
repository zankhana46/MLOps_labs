# Ray Parallel Training Demo - Wine Quality Classification

## 📋 Overview

This project demonstrates how to use **Ray** for parallel machine learning model training. It compares sequential training (one model at a time) versus parallel training (multiple models simultaneously) using Random Forest classifiers on the Wine Quality dataset.

## 🎯 Objectives

- Learn how to use Ray for distributed computing
- Understand the performance benefits of parallel training
- Compare sequential vs parallel execution times
- Train multiple Random Forest models with different hyperparameters

## 📊 Dataset

**Wine Quality Dataset**
- Source: OpenML (scikit-learn built-in)
- Samples: 6,497 wine samples
- Features: 11 physicochemical properties
- Task: Binary classification (good wine vs bad wine)
- Target: Wine quality ≥ 6 = good (1), otherwise bad (0)

## 🚀 Results

### Performance Comparison

| Method | Wall Time | Speedup |
|--------|-----------|---------|
| **Sequential Training** | 1.58 seconds | 1x (baseline) |
| **Parallel Training (Ray)** | 0.491 seconds | **3.2x faster** |

### Key Findings

- ✅ **3.2x speedup** achieved with Ray parallelization
- ✅ Trained 10 Random Forest models with 50-230 trees each
- ✅ Best accuracy: **80.63%** with 50 trees
- ✅ Minimal Ray overhead (~200ms for coordination)

## 🛠️ Technologies Used

- **Python 3.x**
- **Ray** - Distributed computing framework
- **scikit-learn** - Machine learning library
- **pandas** - Data manipulation
- **matplotlib** - Visualization

## 📦 Installation

```bash
pip install ray scikit-learn pandas matplotlib
```

## 🔧 How to Run

1. Open Jupyter Notebook or JupyterLab
2. Create a new notebook
3. Copy and paste each cell from the provided code
4. Run cells sequentially from Cell 1 to Cell 16
5. Observe the timing differences between sequential and parallel training

## 🧠 Key Ray Concepts

### 1. Object Store
```python
X_train_ref = ray.put(X_train)  # Store data once, accessible by all workers
```

### 2. Remote Functions
```python
@ray.remote
def train_model(...):  # Marks function for parallel execution
    # Training code here
```

### 3. Asynchronous Execution
```python
results_ref = [train_model.remote(...) for i in range(10)]  # Launch all tasks
results = ray.get(results_ref)  # Wait for completion and collect results
```

## 📈 Model Performance Insights

The experiment revealed interesting patterns:
- **50 trees**: Best accuracy (80.63%)
- **70-110 trees**: Slight drop in accuracy (~79.6%)
- **130-190 trees**: Recovery to ~80%
- **210-230 trees**: Diminishing returns (~79.4%)

**Lesson**: More trees don't always mean better performance. Optimal hyperparameters vary by dataset.

## 💡 Real-World Applications

This Ray pattern scales to:
- **Hyperparameter tuning**: Train 100s of models with different parameters
- **Cross-validation**: Parallel fold processing
- **Ensemble methods**: Train multiple models simultaneously
- **Feature engineering**: Test multiple feature sets in parallel
- **Model comparison**: Evaluate different algorithms concurrently

### Scaling Example

| Models | Sequential Time | Parallel Time (3-4 cores) | Time Saved |
|--------|----------------|---------------------------|------------|
| 10 | 1.58s | 0.49s | 1.09s (69% faster) |
| 100 | 15.8s | 4.9s | 10.9s (69% faster) |
| 1000 | 158s (2.6 min) | 49s | 109s (69% faster) |

