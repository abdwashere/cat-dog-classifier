# Cat vs Dog Image Classifier

A binary image classification model that distinguishes between cats and dogs, trained using machine learning techniques and deployed on Hugging Face Spaces.

## Live Demo

🚀 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/abdwashere/cat-dog-classifier) <!-- replace with your actual link -->

## Features

- Binary image classification (cat vs dog)
- Model trained and fine-tuned on labeled image dataset
- Deployed as an interactive web app on Hugging Face Spaces
- Supports image upload for real-time prediction

## Tech Stack

| Component | Tool |
|---|---|
| Framework | TensorFlow / Scikit-learn |
| Deployment | Hugging Face Spaces |
| Language | Python |

## Model Architecture

- Convolutional Neural Network (CNN) for feature extraction
- Fine-tuned on cat/dog image dataset
- Binary output: Cat (0) or Dog (1)

## Getting Started

```bash
git clone https://github.com/abdwashere/cat-dog-classifier
cd cat-dog-classifier
pip install -r requirements.txt
python predict.py --image your_image.jpg
```

## Results

98% accuracy using a fine-tuned ResNet18 model <br>
Transfer learning with ResNet18 (+28% accuracy over custom CNN)<br>


<!-- Fill in your actual results -->

## What I Learned

- CNN architecture design and tuning
- Image preprocessing and augmentation
- Model deployment workflow with Hugging Face Spaces
- Trade-offs between model size and accuracy
