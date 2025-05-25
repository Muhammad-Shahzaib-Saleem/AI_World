# 📄 File Similarity Checker

A comprehensive Streamlit application that analyzes and compares the content similarity between two files using advanced cosine similarity with TF-IDF vectorization.

## ✨ Features

### Core Functionality
- **Multi-format Support**: TXT, PDF, DOCX, CSV files
- **Advanced Similarity Analysis**: Cosine similarity with TF-IDF vectorization
- **Interactive Visualizations**: Gauge charts, bar charts, and statistical comparisons
- **Real-time Processing**: Instant analysis with progress indicators

### Analysis Capabilities
- **Similarity Score**: Precise cosine similarity calculation (0.0 - 1.0)
- **Text Statistics**: Character, word, sentence, and unique word counts
- **Common Words Analysis**: Frequency analysis of shared vocabulary
- **Word Clouds**: Visual representation of text content (optional)
- **Text Preview**: Side-by-side content comparison

### User Interface
- **Modern Design**: Clean, responsive interface with gradient styling
- **Sidebar Controls**: Easy file upload and analysis options
- **Interactive Charts**: Plotly-powered visualizations
- **Customizable Settings**: Adjustable preprocessing parameters

## 📁 Supported File Types

- **Text Files** (.txt): Plain text documents
- **PDF Documents** (.pdf): Portable Document Format
- **Word Documents** (.docx, .doc): Microsoft Word files
- **CSV Files** (.csv): Comma-separated values

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit application:
```bash
streamlit run app.py
```

## How It Works

1. **Upload Files**: Select two text files using the file uploaders
2. **Preview Content**: Optionally preview the content of uploaded files
3. **Calculate Similarity**: Click the "Calculate Similarity" button
4. **View Results**: See the cosine similarity score, similarity level, and interpretation

## Similarity Interpretation

- **0.8 - 1.0**: Very High similarity (files are nearly identical)
- **0.6 - 0.8**: High similarity (significant common content)
- **0.4 - 0.6**: Medium similarity (some common elements)
- **0.2 - 0.4**: Low similarity (few common elements)
- **0.0 - 0.2**: Very Low similarity (quite different content)

## Technical Details

The application uses:
- **TF-IDF Vectorization**: Converts text to numerical vectors
- **Cosine Similarity**: Measures the angle between vectors
- **Streamlit**: Provides the web interface
- **Scikit-learn**: Handles the machine learning computations

## Example Use Cases

- Compare different versions of documents
- Check for plagiarism or duplicate content
- Analyze similarity between code files
- Compare research papers or articles
- Evaluate content variations