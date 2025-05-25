# File Similarity Checker

A Streamlit web application that calculates the similarity between two text files using cosine similarity.

## Features

- 📄 Upload two text files for comparison
- 🔍 Calculate cosine similarity using TF-IDF vectors
- 📊 Visual similarity metrics and progress bars
- 📈 Detailed interpretation of similarity scores
- 📋 File statistics (character, word, and line counts)
- 🎨 Clean and intuitive user interface

## Supported File Types

- `.txt` - Plain text files
- `.md` - Markdown files
- `.py` - Python files
- `.js` - JavaScript files
- `.html` - HTML files
- `.css` - CSS files
- `.json` - JSON files
- `.csv` - CSV files

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