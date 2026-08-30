# 📄 AI Resume Analyzer

An intelligent Streamlit dashboard that leverages Natural Language Processing (NLP) and Machine Learning to analyze resumes, predict job categories, and evaluate ATS compatibility.
<p align="center"> <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" alt="Python"/> <img src="https://img.shields.io/badge/Streamlit-1.29%2B-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit"/> <img src="https://img.shields.io/badge/scikit--learn-1.5.0-orange?style=for-the-badge&logo=scikit-learn" alt="scikit-learn"/> <img src="https://img.shields.io/badge/NLTK-3.8.1-green?style=for-the-badge&logo=nltk" alt="NLTK"/> <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License"/> </p>

---

## ✨ Features

- **🔍 Job Category Prediction**: Uses TF-IDF vectorization and Vector Ensemble to predict the best-fit job category from a resume.
- **📊 ATS Compatibility Scoring**: Evaluates resumes against job descriptions or performs general resume health checks.
- **🎯 Resume Parsing & Analysis**: Extracts and analyzes key information including skills, experience, and education.
- **📈 Interactive Visualizations**: View TF-IDF scores, similarity metrics, and category predictions through dynamic charts.
- **📝 Side-by-Side Comparison**: See original vs. cleaned text with tokenization statistics.
- **🔬 EDA Notebook**: Comprehensive exploratory data analysis of resume datasets.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML/NLP**: scikit-learn, NLTK, pandas, NumPy
- **Visualization**: Plotly, matplotlib, seaborn
- **Document Processing**: pdfplumber, python-docx

---

## 🎯 How It Works

### NLP Pipeline

1. **Text Extraction** - Extracts text from PDF, DOCX, or TXT files.
2. **Cleaning** - Removes URLs, mentions, hashtags, and special characters.
3. **Tokenization** - Splits text into individual tokens.
4. **Stopword Removal** - Removes common stopwords.
5. **Lemmatization** - Reduces words to their base form.
6. **TF-IDF Vectorization** - Converts text to numerical features.
7. **ML Prediction** - Uses Vector Ensemble for classification.

### ATS Scoring

The system evaluates resumes on multiple dimensions:

- **Content Analysis**: Skills matching and keyword density.
- **Formatting**: Contact info, section headers, and bullet points.
- **Action Verbs**: Presence of strong action verbs.
- **Quantification**: Use of metrics and numbers.
- **Experience Match**: Years of experience alignment.

---

## 📊 Dataset

The model is trained on a dataset of **12,000 resumes** across **25 job categories**, including:

- Java Developer, DevOps Engineer, Data Science
- Python Developer, Web Designing, HR
- Blockchain, Operations Manager, Testing

And many more...

---

## 🎨 UI Features

- **Sidebar**: Model info, pipeline stages, and ATS scoring overview.
- **Upload Area**: Drag-and-drop resume upload with job description option.
- **Results View**: Category prediction, confidence scores, and ATS compatibility.
- **Visualizations**: Category distribution, similarity scores, and TF-IDF weights.
- **Pipeline View**: Visual representation of the NLP pipeline stages.





<p align="center">
  <img src="Screenshot 2026-08-29 000633.png" alt="Alt text" width="700"/>
</p>
<p align="center">
  <img src="Screenshot 2026-08-29 000707.png" alt="Alt text" width="700"/>
</p>
<p align="center">
  <img src="Screenshot 2026-08-29 000724.png" alt="Alt text" width="700"/>
</p>
<p align="center">
  <img src="Screenshot 2026-08-29 000738.png" alt="Alt text" width="700"/>
</p>
<p align="center">
  <img src="Screenshot 2026-08-29 000756.png" alt="Alt text" width="700"/>
</p>

  
