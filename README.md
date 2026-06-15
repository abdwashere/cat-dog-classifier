A production-grade MLOps project that classifies cats and dogs with 98% accuracy using a fine-tuned ResNet18 model, complete with experiment tracking, REST API serving, continuous training, and automated redeployment.


Live Demo: https://huggingface.co/spaces/abdwashere/cat-dog-classifier


How to Run it locally:<br>
  1. Clone the repo:<br>
      git clone https://github.com/abdwashere/cat-dog-classifier.git <br>
      cd cat-dog-classifier<br>
     
  2. Set up the enviorment: <br>
      python -m venv venv<br>
      venv\Scripts\activate      # Windows<br>
      pip install torch torchvision fastapi uvicorn pillow pyyaml mlflow scikit-learn gradio numpy<br>
     
  3. Update params.yaml with your data paths:<br>
      data:<br>
          cats_folder: "path/to/cats"<br>
          dogs_folder: "path/to/dogs"<br>
  
  4. Train the model:<br>
      python src/train.py<br>

  5. Start the API:<br>
      uvicorn api.main:app --reload

  6. Test at http://127.0.0.1:8000/docs<br>
<br>
<br>

What I Learned

End-to-end MLOps pipeline from training to production.<br>
Transfer learning with ResNet18 (+28% accuracy over custom CNN)<br>
Model serving with FastAPI<br>
Experiment tracking with MLflow<br>
Continuous training with GitHub Actions<br>
Model monitoring and drift detection<br>
Deploying ML apps on HuggingFace Spaces<br>
