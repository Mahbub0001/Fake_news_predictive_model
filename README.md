# 📰 Fake News Detection System

A sophisticated machine learning application designed to detect and classify news articles as either **Fake** or **True** with exceptional accuracy. This project leverages advanced NLP techniques and ensemble learning methods to combat misinformation in digital media.

## 🎯 Project Overview

This fake news detection system achieves **99.69% accuracy** using an XGBoost classifier trained on a comprehensive dataset of real and fake news articles. The application features both a Jupyter notebook for model development and training, and an interactive web interface built with Gradio for real-time predictions.

## 📊 Dataset

The project utilizes two comprehensive datasets:

- **True.csv**: 21,417 legitimate news articles from reputable sources
- **Fake.csv**: 23,481 fake news articles
- **Total**: 44,898 articles for training and evaluation

### Dataset Features
- `title`: News article headline
- `text`: Full article content
- `subject`: News category (politics, world news, etc.)
- `date`: Publication date
- `label`: Binary classification (0=Fake, 1=True)

## 🚀 Key Features

### Model Performance
- **Accuracy**: 99.69%
- **Precision**: 100% (both classes)
- **Recall**: 100% (both classes)
- **F1-Score**: 100% (both classes)

### Model Comparison
| Model | Accuracy |
|-------|----------|
| XGBoost | 99.69% |
| Gradient Boosting | 99.50% |
| Linear SVC | 99.23% |
| Support Vector Classifier | 99.07% |
| Random Forest | 98.84% |
| Logistic Regression | 98.16% |

## 🛠️ Technology Stack

- **Machine Learning**: Scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy
- **Text Processing**: TF-IDF Vectorization
- **Web Interface**: Gradio
- **Visualization**: Matplotlib, Seaborn
- **Model Persistence**: Pickle

## 📁 Project Structure

```
Fake_news/
├── Fake_News_Detection.ipynb    # Jupyter notebook for model development
├── app.py                       # Gradio web application
├── model.pkl                    # Trained XGBoost model
├── Fake.csv                     # Fake news dataset
├── True.csv                     # Real news dataset
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🧠 Model Architecture

### Text Processing Pipeline
1. **TF-IDF Vectorization**: Converts text to numerical features
   - Stop words removal (English)
   - Maximum document frequency: 0.7
   - Vocabulary size: 121,689 features

2. **Feature Engineering**: Sparse matrix representation with 6,848,207 stored elements

3. **Classification**: XGBoost ensemble learning with optimized hyperparameters

## 🌐 Web Application

The Gradio-based web interface provides:
- **Real-time Prediction**: Instant classification of news text
- **Confidence Scores**: Probability distribution for Fake/True classes
- **User-Friendly Interface**: Clean, intuitive design
- **Example Inputs**: Pre-loaded sample texts for testing

### Features
- Text input area with multi-line support
- Probability visualization
- Confidence percentage display
- Responsive design for all devices

## 📋 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Fake_news
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
python app.py
```

The web interface will be available at `http://localhost:7860`

## 🔧 Usage

### Web Interface
1. Open the application in your browser
2. Paste or type news text in the input area
3. Click "Predict" to get classification results
4. View confidence scores and probability distribution

### Programmatic Usage
```python
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare text (using same vectorizer as training)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# ... (fit vectorizer on training data)

# Make prediction
text = "Your news text here"
X = vectorizer.transform([text])
prediction = model.predict(X)[0]
probability = model.predict_proba(X)[0]

print(f"Prediction: {'True' if prediction == 1 else 'Fake'}")
print(f"Confidence: {max(probability)*100:.2f}%")
```

## 📈 Model Evaluation

### Confusion Matrix
The model demonstrates perfect classification with:
- **True Positives**: 5,362
- **True Negatives**: 5,863
- **False Positives**: 0
- **False Negatives**: 0

### ROC Curve
- **AUC Score**: 1.00 (Perfect classification)
- **ROC Curve**: Optimal trade-off between TPR and FPR

## 🔬 Model Development Process

1. **Data Preprocessing**: Combined and shuffled datasets
2. **Feature Extraction**: TF-IDF vectorization with optimized parameters
3. **Model Selection**: Evaluated 6 different algorithms
4. **Hyperparameter Tuning**: Selected XGBoost as best performer
5. **Validation**: 75/25 train-test split with random_state=42
6. **Evaluation**: Comprehensive metrics and visualizations

## 🎨 Visualizations

The project includes detailed visualizations:
- **Confusion Matrix**: Heatmap of classification performance
- **ROC Curve**: Receiver Operating Characteristic analysis
- **Model Comparison**: Bar chart of accuracy scores

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Future Enhancements

- [ ] Multi-language support
- [ ] Real-time news feed integration
- [ ] Advanced text preprocessing (lemmatization, stemming)
- [ ] Deep learning models (BERT, RoBERTa)
- [ ] API endpoint for integration
- [ ] Batch processing capabilities
- [ ] Model explainability features

## ⚠️ Limitations

- **Language**: Currently optimized for English text only
- **Context**: May struggle with highly specialized domains
- **Evolving Content**: Model performance may degrade with new types of misinformation
- **Bias**: Training data may contain inherent biases

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Mahbub Ul Alam Bhuiyan**
- Machine Learning Engineer
- Specialized in NLP and Text Classification

## 🙏 Acknowledgments

- Dataset providers for comprehensive news collections
- Open-source community for excellent ML libraries
- Scikit-learn and XGBoost teams for powerful algorithms

## 📞 Contact

For questions, suggestions, or collaborations:
- **Email**: [nibirbhuiyan18@gmail.com](nibirbhuiyan18@gmail.com)
- **LinkedIn**: [linkedin.com/in/mahbub-ul-alam-bhuiyan-289bb8294](linkedin.com/in/mahbub-ul-alam-bhuiyan-289bb8294)
- **GitHub**: [https://github.com/Mahbub0001](https://github.com/Mahbub0001)


**⚡ Built with passion for combating misinformation and promoting media literacy**
