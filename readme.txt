Title: Brain Tumor Classification using CNN-Transformer Hybrid Models with Visualization

Description:
This project focuses on classifying brain tumors into four categories: Glioma, Meningioma, Pituitary, and No Tumor. The model uses two deep learning architectures - ResNet50 and EfficientNetB0 - both enhanced with SE blocks and Transformer encoder layers for better feature extraction and global attention.

Main Features:
- Data loading and preprocessing with augmentation for robustness.
- EDA: class distribution, image size stats, pixel stats, and image samples.
- Two CNN-Transformer Hybrid models:
    1. ResNet50-based Hybrid
    2. EfficientNetB0-based Hybrid
- Training with weighted cross-entropy loss to address class imbalance.
- Evaluation with Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.
- Grad-CAM and Saliency Map visualization for model interpretability.
- Test-Time Augmentation (TTA) for better predictions.
- Gradio Web Interface for user-friendly predictions and heatmap visualization.

Folder Structure:
/kaggle/input/brain-tumerr/
│
├── brain tumer/
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── no_tumor/
│   │   └── pituitary/
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── no_tumor/
│       └── pituitary/
│
├── resnet50-11ad3fa6.pth          # Pretrained weights for ResNet50
└── efficientnet_b0.pth            # Pretrained weights for EfficientNetB0

Files:
- code.ipynb                   # The full code (training, evaluation, visualization)
- app.py                       # UI frontend part using Gradio
- hybrid_models.pth            # Saved model weights for ResNet-Hybrid and EffNet-Hybrid
- requirements.txt             # Required libraries (PyTorch, torchvision, matplotlib, seaborn, scikit-learn, Gradio, etc.)
- Testing Images               # for giving input use these images 
- code                         # In code file run app.py 
- run                          # run this run.sh in the code file only 
Dependencies:
torch>=2.0  
torchvision>=0.15  
tqdm  
numpy  
matplotlib  
seaborn  
scikit-learn  
Pillow  
opencv-python  
gradio>=4.0  

You can install all dependencies with:
pip install torch torchvision tqdm numpy matplotlib seaborn scikit-learn Pillow opencv-python gradio

How to Run:
1. Ensure that all dependencies are installed (see requirements.txt).
2. To train and evaluate the models, run the Jupyter notebook:
   - Open `code.ipynb` and execute all cells.
3. To launch the interactive frontend:
   - Run the command below in terminal or command prompt:
     ```
     python app.py
     ```
4. Use the Gradio UI in your browser to upload a brain MRI image and view:
   - Predicted probabilities
   - Grad-CAM overlays (ResNet & EffNet)
   - Saliency maps (ResNet & EffNet)

Notes:
- Both Grad-CAM and Saliency maps are computed per prediction to provide interpretability.
- Preprocessing includes resizing to 224x224 and normalization.
- TTA aggregates predictions from multiple transformed inputs for robustness.
- All code is written in PyTorch and can run on GPU for faster execution.

Credits:
- Pretrained weights for ResNet50 and EfficientNetB0 from torchvision model zoo.
- Gradio library for creating the interactive web interface.
