🚀 Skin Cancer Detection Web App

A Machine Learning + Flask Web Application for Skin Lesion Classification

📌 Overview

This project is an end-to-end machine learning web application designed to classify skin lesion images as Benign or Malignant using a custom-built Convolutional Neural Network (CNN) trained in PyTorch. The app provides an interactive and user-friendly interface powered by Flask, Tailwind CSS, and Chart.js, and includes advanced features like image preview, confidence visualization, and probability charts.

⚠️ Disclaimer: This project is strictly for educational and demonstration purposes. It is not a medical diagnostic tool and must not be used for real clinical decisions.

✨ Features

🧠 Custom CNN model built with PyTorch

🖼️ Image upload & preview through the browser

📊 Class probability bar chart (Chart.js)

📉 Dynamic confidence bar reflecting prediction strength

💬 Detailed explanation section for interpreting model results

⚠️ Medical disclaimer for safety

🎨 Modern responsive UI using Tailwind CSS

🔐 Secure preprocessing pipeline with PIL image handling

🔧 Virtual environment setup (.venv) for reproducible execution

🧠 Model Architecture

The CNN includes:

3 convolutional layers

Max pooling after each conv

Flatten layer

6 fully connected layers

LogSoftmax output

Optimized for binary classification (Benign vs Malignant).

🛠️ Tech Stack

Backend: Python, Flask, PyTorch Frontend: Tailwind CSS, HTML, Jinja2, Chart.js Image Handling: PIL, Base64 Encoding Environment: .venv, Pip, VS Code

🖥️ UI Preview

The interface includes:

❇️ Prediction result

🎯 Confidence percentage

📊 Probability chart

🟩 Confidence bar

🖼️ Uploaded image preview

Everything renders automatically after upload.

🧪 Dataset

This project uses a curated set of dermoscopic images (Kaggle/ISIC-style datasets). You may replace the dataset or retrain the model with additional images to improve performance.

🔥 Future Enhancements

Grad-CAM visualizations for heatmap explainability

Multi-class lesion support

Model retraining with larger datasets

Deployment to Render / Railway / Azure
