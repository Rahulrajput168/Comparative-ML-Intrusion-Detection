# 🛡️ Comparative Analysis of Machine Learning Algorithms for Intrusion Detection

### A Machine Learning and Deep Learning Approach to Network Security

<p align="center">
  <strong>Detect. Analyze. Compare. Secure.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-Model%20Comparison-orange" alt="Machine Learning"/>
  <img src="https://img.shields.io/badge/Deep%20Learning-MLP%20%7C%20Transformer-purple" alt="Deep Learning"/>
  <img src="https://img.shields.io/badge/Dataset-NSL--KDD-green" alt="NSL-KDD"/>
</p>

---

## 🚀 Project Overview

**Comparative Analysis of Machine Learning Algorithms for Intrusion Detection** is a cybersecurity project that evaluates multiple machine learning and deep learning algorithms for identifying potentially malicious network traffic.

The project uses the **NSL-KDD dataset** to classify network connections into two categories:

- **Normal Traffic** — network connections classified as normal.
- **Attack Traffic** — network connections classified as potentially malicious.

The system combines data preprocessing, feature selection, model training, performance evaluation, and interactive visualization in a Streamlit dashboard.

It enables users to compare different algorithms, inspect model evaluation results, and perform predictions on individual records or uploaded CSV datasets.

## 🌐 Live Demo

**Deployed Application:** [Open the Intrusion Detection Dashboard](YOUR_DEPLOYED_STREAMLIT_URL)

> Replace `YOUR_DEPLOYED_STREAMLIT_URL` with your actual public deployment URL once your application is deployed.

**Source Code:** [View the GitHub Repository](YOUR_GITHUB_REPOSITORY_URL)

> Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

## ✨ Key Features

- **Interactive Dashboard:** Overview of the dataset, models, and evaluation results.
- **Multiple Algorithm Comparison:** Compare traditional machine learning, boosting, and deep learning approaches.
- **Feature Selection:** Evaluate an XGBoost model using a reduced set of 50 selected features.
- **Confusion Matrix Visualization:** Inspect classification performance for the evaluated models.
- **Single Record Prediction:** Submit an individual network record for classification.
- **CSV Batch Prediction:** Upload a CSV file to generate predictions for multiple records.
- **XGBoost Analysis:** Explore the performance of full-feature and feature-selected XGBoost models.
- **Performance Metrics:** Review accuracy, precision, recall, and F1-score.
- **Saved Models:** Use trained model artifacts for prediction without retraining every time.

## 🧠 Machine Learning Models

The project includes nine model configurations:

| No. | Model | Category |
|---:|---|---|
| 1 | Logistic Regression | Traditional ML |
| 2 | Decision Tree | Traditional ML |
| 3 | Random Forest | Ensemble Learning |
| 4 | K-Nearest Neighbors (KNN) | Traditional ML |
| 5 | Support Vector Machine (SVM) | Traditional ML |
| 6 | XGBoost | Gradient Boosting |
| 7 | Feature-Selected XGBoost | Feature Selection + Boosting |
| 8 | Multi-Layer Perceptron (MLP) | Deep Learning |
| 9 | Tabular Transformer | Deep Learning / Attention |

The models are compared using a consistent evaluation workflow. The dashboard presents the available experimental results to support algorithm comparison.

## 📊 Dataset Information

The project uses the **NSL-KDD intrusion detection dataset**, which contains network connection records for studying normal and malicious traffic.

| Dataset property | Details |
|---|---|
| Dataset | NSL-KDD |
| Training records | 125,973 |
| Testing records | 22,544 |
| Processed features | 122 |
| Selected features for feature-selected model | 50 |
| Classification type | Binary classification |
| Output classes | Normal Traffic, Attack Traffic |

### Data Preprocessing

The preprocessing pipeline includes:

1. Loading the training and testing datasets.
2. Removing the `difficulty` field.
3. Converting the target labels into binary classes.
4. Encoding categorical network attributes.
5. Aligning training and testing feature columns.
6. Standardizing features where required by the model pipeline.
7. Preparing processed data for model training and evaluation.

The target classes are represented as:

- `0` — Normal Traffic
- `1` — Attack Traffic

## 📈 Experimental Results

The models are evaluated using the following performance metrics:

- **Accuracy:** Overall proportion of correctly classified records.
- **Precision:** Proportion of predicted positive cases that are positive.
- **Recall:** Proportion of actual positive cases identified by the model.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Summary of actual versus predicted classifications.

### Transformer Model — Test Results

| Metric | Result |
|---|---:|
| Accuracy | 80.37% |
| Precision | 92.62% |
| Recall | 71.19% |
| F1-score | 80.50% |

These results were obtained on the project's NSL-KDD test dataset. Results may vary across algorithms and configurations. Refer to the project's evaluation reports and dashboard for the corresponding model comparison results.

## 🏗️ System Architecture

```text
             NSL-KDD Dataset
                    |
                    v
          Data Loading & Inspection
                    |
                    v
             Data Preprocessing
                    |
                    v
        Feature Encoding & Scaling
                    |
                    v
             Feature Selection
                    |
                    v
       Machine Learning / Deep Learning
                    |
                    v
          Model Evaluation & Testing
                    |
                    v
            Saved Model Artifacts
                    |
                    v
          Streamlit Web Dashboard
                    |
          +---------+----------+
          |         |          |
          v         v          v
       Compare   Single      CSV Batch
       Models   Prediction   Prediction
```

## 🖥️ Dashboard Modules

### 1. Dashboard
Displays an overview of the project, dataset statistics, model information, and evaluation highlights.

### 2. Model Comparison
Provides a comparative view of the implemented algorithms using their available evaluation metrics.

### 3. Confusion Matrix
Visualizes classification outcomes for supported models.

### 4. Single Prediction
Allows a user to submit one network record and view the predicted class.

### 5. CSV Prediction
Supports batch prediction by uploading a CSV file containing compatible input features.

### 6. XGBoost Analysis
Provides analysis of the XGBoost and feature-selected XGBoost configurations.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computing |
| Scikit-learn | Traditional ML algorithms and preprocessing |
| XGBoost | Gradient-boosting model |
| PyTorch | Transformer-based deep learning |
| Matplotlib / Seaborn / Plotly | Data visualization, where used |
| Streamlit | Interactive web dashboard |
| Joblib | Saving and loading supported model artifacts |

## 📂 Project Structure

The project is organized into components for data processing, model development, evaluation, and dashboard functionality.

```text
Comparative-ML-Intrusion-Detection/
│
├── app/
│   └── streamlit_app.py
│
├── dataset/
│   ├── KDDTrain+.txt
│   ├── KDDTest+.txt
│   └── load_dataset.py
│
├── src/
│   └── train_transformer.py
│
├── processed_data/
│
├── saved_models/
│
├── reports/
│   ├── figures/
│   └── evaluation_results.csv
│
├── config.py
├── requirements.txt
└── README.md
```

*Note: This is a representative structure. Retain the actual filenames and folders present in your repository when publishing the README.*

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.12 recommended
- Git
- pip
- The project dataset and required model artifacts

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Comparative-ML-Intrusion-Detection
```

Replace the repository URL with your actual GitHub repository URL.

### Step 2: Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, use Command Prompt and run:

```bat
venv\Scripts\activate.bat
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure that `requirements.txt` includes the packages required by the actual project, including Streamlit, Pandas, NumPy, Scikit-learn, XGBoost, PyTorch, and other imported libraries.

### Step 4: Run the Dashboard

```bash
streamlit run app/streamlit_app.py
```

Open the local address shown in the terminal, typically:

```text
http://localhost:8501
```

Use the dashboard navigation to explore model comparison, confusion matrices, single-record predictions, CSV predictions, and XGBoost analysis.

**Important:** Use the existing trained model files and processed data required by the application. Retraining is not necessary just to launch the dashboard.

## ☁️ Deployment

The Streamlit dashboard can be deployed using a compatible hosting platform such as Streamlit Community Cloud.

General deployment steps:

1. Push the project source code to a GitHub repository.
2. Confirm that `requirements.txt` contains the necessary dependencies.
3. Ensure that the application entry point is `app/streamlit_app.py`.
4. Configure the hosting service to use the correct Python environment.
5. Ensure all required model artifacts and permitted dataset files are accessible.
6. Deploy the application and test its dashboard pages.
7. Add the resulting public URL to the **Live Demo** section of this README.

Large model files or datasets may require Git LFS or another appropriate hosting method. Do not upload secrets, private credentials, or restricted data to a public repository.

## 🔒 Limitations

- Model performance depends on the dataset, preprocessing pipeline, and model configuration.
- The current system performs binary classification into normal and attack traffic.
- The NSL-KDD dataset may not represent all modern real-world network attacks.
- Prediction quality should be assessed using the reported test metrics rather than training performance alone.
- The application is an academic research prototype and should not be treated as a replacement for a production-grade network security system.

## 🔮 Future Scope

Possible extensions include:

- Evaluating additional modern intrusion detection datasets.
- Investigating methods to improve attack detection and generalization.
- Exploring explainable AI techniques for understanding model predictions.
- Extending classification to individual attack categories.
- Evaluating deployment performance under realistic traffic conditions.
- Improving monitoring, scalability, and integration with network security tools.

## 🎯 Project Objectives

- Compare traditional machine learning and deep learning algorithms for intrusion detection.
- Study the impact of feature selection on model performance.
- Evaluate models using standard classification metrics.
- Provide an interactive dashboard for analyzing results.
- Support individual and batch network traffic classification.

## 👨‍💻 Author

**Rahul Singh**  
B.Tech Computer Science and Engineering  
Specialization: Data Science and Big Data Tools

**Project:** Comparative Analysis of Machine Learning Algorithms for Intrusion Detection

## 📜 License

Choose and add a license before distributing the project publicly. If no license is specified, the repository's default copyright restrictions still apply.

---

<p align="center">
  <strong>Built with Python, Machine Learning, Deep Learning, and Streamlit.</strong>
</p>
