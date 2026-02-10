Jaguar Re-Identification with Deep Metric Learning

This repository implements an end-to-end Jaguar Re-Identification (ReID) system using deep metric learning, ConvNeXt, and ArcFace loss.
The pipeline is designed for high-accuracy wildlife re-identification, enabling robust matching between query and gallery images under varying poses, lighting, and backgrounds.

🚀 Key Features

✅ ConvNeXt backbone (ImageNet pretrained via timm)

✅ ArcFace loss for discriminative embedding learning

✅ GeM pooling for stronger spatial feature aggregation

✅ Test-Time Augmentation (TTA) with horizontal flips

✅ Cosine similarity–based retrieval

✅ Interactive & static visualizations (PCA, t-SNE, heatmaps)

✅ Submission-ready Kaggle pipeline

🧠 Model Architecture
Input Image (224×224)
        ↓
ConvNeXt-Small Backbone
        ↓
GeM Pooling
        ↓
BatchNorm Embedding Head
        ↓
ArcFace Margin Classifier (Training Only)
        ↓
L2-Normalized Embeddings (Inference)

📂 Project Structure
├── train.csv
├── test.csv
├── train/
│   └── train/          # Training images
├── test/
│   └── test/           # Test images
├── jaguar_best_model.pth
├── submission.csv
├── train.py / notebook.ipynb
└── README.md

⚙️ Configuration Highlights
Parameter	Value
Backbone	convnext_small.fb_in1k
Image Size	224 × 224
Embedding Dim	768
Loss	ArcFace
Optimizer	AdamW
Scheduler	CosineAnnealingLR
Batch Size	32
Epochs	8
Mixed Precision	✅
🧪 Training Strategy

Strong data augmentation

Random affine

Color jitter

Horizontal flip

Random erasing

Metric learning objective

ArcFace margin (m=0.5, s=30)

Mixed-precision training

Faster convergence

Lower GPU memory usage

🔍 Inference & Similarity Scoring

Extract embeddings for query + gallery images

Apply TTA averaging

L2-normalize embeddings

Compute cosine similarity matrix

Generate submission with similarity scores clipped to [0, 1]

📊 Visualizations Included
Static

📈 Training loss curve

📉 Similarity score distribution

🧭 PCA & t-SNE embedding projections

🔥 Cosine similarity & distance heatmaps

🖼️ Top similar image pairs

Interactive (Plotly)

🧠 Interactive PCA / t-SNE scatter plots

🧩 Hover-enabled similarity heatmaps

🖱️ Image-level exploration

🛠️ Tech Stack

Languages

Python

Deep Learning

PyTorch

TIMM

Torch AMP (mixed precision)

Data & Visualization

NumPy, Pandas

Matplotlib, Seaborn

Plotly

ML Utilities

scikit-learn

PIL, torchvision

📈 Results & Observations

ArcFace significantly improves inter-class separation

GeM pooling outperforms standard average pooling

TTA improves retrieval stability

Embeddings show clear clustering in PCA/t-SNE space

🏁 How to Run
pip install timm torch torchvision plotly seaborn scikit-learn
python train.py


Or run directly in Kaggle Notebook (GPU recommended).

📌 Future Improvements

🔄 k-reciprocal re-ranking

🧬 Hard sample mining

🧠 Larger backbones (ConvNeXt-B, ViT)

📏 Cross-validation with fold ensembling
