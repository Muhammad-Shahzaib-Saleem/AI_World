import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import re
import docx
import PyPDF2
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def extract_text_from_file(uploaded_file):
    """Extract text content from various file formats"""
    try:
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        if file_extension == '.txt':
            # Text file
            content = str(uploaded_file.read(), "utf-8")
            return content
        
        elif file_extension == '.pdf':
            # PDF file
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        
        elif file_extension in ['.docx', '.doc']:
            # Word document
            doc = docx.Document(uploaded_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        
        elif file_extension == '.csv':
            # CSV file - combine all text content
            df = pd.read_csv(uploaded_file)
            text = df.to_string()
            return text
        
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None
            
    except Exception as e:
        st.error(f"Error reading file {uploaded_file.name}: {str(e)}")
        return None

def preprocess_text(text):
    """Clean and preprocess text for better similarity calculation"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove very short words (less than 3 characters)
    words = text.split()
    words = [word for word in words if len(word) >= 3]
    
    return ' '.join(words)

def calculate_cosine_similarity(text1, text2):
    """Calculate cosine similarity between two texts using TF-IDF"""
    if not text1 or not text2:
        return 0.0
    
    # Create TF-IDF vectorizer with more flexible parameters
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words=None,  # Don't remove stop words to preserve more content
        ngram_range=(1, 1),  # Use only unigrams for better compatibility
        min_df=1,
        max_df=1.0,
        lowercase=True,
        token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only words with 2+ letters
    )
    
    try:
        # Fit and transform the texts
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        
        # Check if we have any features
        if tfidf_matrix.shape[1] == 0:
            st.warning("No features found after vectorization. Texts might be too short or dissimilar.")
            return 0.0
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        similarity_score = similarity_matrix[0][1]
        
        # Ensure the score is between 0 and 1
        similarity_score = max(0.0, min(1.0, similarity_score))
        
        return similarity_score
    
    except Exception as e:
        st.error(f"Error calculating similarity: {str(e)}")
        return 0.0

def get_similarity_interpretation(score):
    """Interpret similarity score and return color class and description"""
    if score >= 0.8:
        return "high-similarity", "Very High Similarity", "🟢"
    elif score >= 0.6:
        return "high-similarity", "High Similarity", "🟢"
    elif score >= 0.4:
        return "medium-similarity", "Medium Similarity", "🟡"
    elif score >= 0.2:
        return "medium-similarity", "Low-Medium Similarity", "🟡"
    else:
        return "low-similarity", "Low Similarity", "🔴"

def create_similarity_gauge(score):
    """Create a gauge chart for similarity score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Similarity Score (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 20], 'color': "lightgray"},
                {'range': [20, 40], 'color': "yellow"},
                {'range': [40, 60], 'color': "orange"},
                {'range': [60, 80], 'color': "lightgreen"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def analyze_text_statistics(text1, text2):
    """Analyze and compare text statistics"""
    stats = {
        'File 1': {
            'Characters': len(text1),
            'Words': len(text1.split()),
            'Unique Words': len(set(text1.split())),
            'Sentences': len(re.split(r'[.!?]+', text1))
        },
        'File 2': {
            'Characters': len(text2),
            'Words': len(text2.split()),
            'Unique Words': len(set(text2.split())),
            'Sentences': len(re.split(r'[.!?]+', text2))
        }
    }
    
    return pd.DataFrame(stats).T

def find_common_words(text1, text2, top_n=20):
    """Find common words between two texts"""
    words1 = set(text1.split())
    words2 = set(text2.split())
    common_words = words1.intersection(words2)
    
    # Count frequency of common words in both texts
    word_freq = {}
    for word in common_words:
        freq1 = text1.split().count(word)
        freq2 = text2.split().count(word)
        word_freq[word] = freq1 + freq2
    
    # Sort by frequency and return top N
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:top_n]

def create_word_frequency_chart(common_words):
    """Create a bar chart of common word frequencies"""
    if not common_words:
        return None
    
    words, frequencies = zip(*common_words)
    
    fig = px.bar(
        x=list(frequencies),
        y=list(words),
        orientation='h',
        title="Most Common Words Between Files",
        labels={'x': 'Frequency', 'y': 'Words'}
    )
    
    fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
    return fig

# Main application
def main():
    # Page configuration
    st.set_page_config(
        page_title="📄 File Similarity Checker",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 2rem;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .similarity-score {
            font-size: 2rem;
            font-weight: bold;
            text-align: center;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .high-similarity {
            background-color: #d4edda;
            color: #155724;
            border: 2px solid #c3e6cb;
        }
        .medium-similarity {
            background-color: #fff3cd;
            color: #856404;
            border: 2px solid #ffeaa7;
        }
        .low-similarity {
            background-color: #f8d7da;
            color: #721c24;
            border: 2px solid #f5c6cb;
        }
        .file-info {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #007bff;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📄 File Similarity Checker</h1>', unsafe_allow_html=True)
    st.markdown("### Compare the content similarity between two files using cosine similarity")
    
    # Sidebar for file uploads
    with st.sidebar:
        st.header("📁 Upload Files")
        st.markdown("**Supported formats:** TXT, PDF, DOCX, CSV")
        
        file1 = st.file_uploader(
            "Choose first file",
            type=['txt', 'pdf', 'docx', 'doc', 'csv'],
            key="file1"
        )
        
        file2 = st.file_uploader(
            "Choose second file",
            type=['txt', 'pdf', 'docx', 'doc', 'csv'],
            key="file2"
        )
        
        # Analysis options
        st.header("⚙️ Analysis Options")
        show_statistics = st.checkbox("Show text statistics", value=True)
        show_common_words = st.checkbox("Show common words analysis", value=True)
        show_word_cloud = st.checkbox("Generate word clouds", value=False)
        
        # Preprocessing options
        st.header("🔧 Preprocessing")
        min_word_length = st.slider("Minimum word length", 1, 5, 3)
        max_features = st.slider("Maximum features for TF-IDF", 1000, 10000, 5000)
    
    # Main content area
    if file1 and file2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f'<div class="file-info"><strong>📄 File 1:</strong> {file1.name}</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="file-info"><strong>📄 File 2:</strong> {file2.name}</div>', unsafe_allow_html=True)
        
        # Extract text from files
        with st.spinner("Extracting text from files..."):
            text1 = extract_text_from_file(file1)
            text2 = extract_text_from_file(file2)
        
        if text1 and text2:
            # Preprocess texts
            with st.spinner("Preprocessing texts..."):
                processed_text1 = preprocess_text(text1)
                processed_text2 = preprocess_text(text2)
            
            # Calculate similarity
            with st.spinner("Calculating similarity..."):
                similarity_score = calculate_cosine_similarity(processed_text1, processed_text2)
            
            # Display similarity results
            st.markdown("---")
            st.header("🎯 Similarity Results")
            
            # Create three columns for the results
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                # Similarity score display
                css_class, interpretation, emoji = get_similarity_interpretation(similarity_score)
                st.markdown(f'''
                <div class="similarity-score {css_class}">
                    {emoji} {interpretation}<br>
                    <span style="font-size: 3rem;">{similarity_score:.3f}</span><br>
                    <span style="font-size: 1rem;">({similarity_score*100:.1f}%)</span>
                </div>
                ''', unsafe_allow_html=True)
            
            # Gauge chart
            st.plotly_chart(create_similarity_gauge(similarity_score), use_container_width=True)
            
            # Detailed analysis
            if show_statistics:
                st.markdown("---")
                st.header("📊 Text Statistics")
                
                stats_df = analyze_text_statistics(processed_text1, processed_text2)
                st.dataframe(stats_df, use_container_width=True)
                
                # Create comparison chart
                fig = px.bar(
                    stats_df.reset_index(),
                    x='index',
                    y=['Characters', 'Words', 'Unique Words', 'Sentences'],
                    title="Text Statistics Comparison",
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            if show_common_words:
                st.markdown("---")
                st.header("🔤 Common Words Analysis")
                
                common_words = find_common_words(processed_text1, processed_text2)
                
                if common_words:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Top Common Words")
                        common_df = pd.DataFrame(common_words, columns=['Word', 'Total Frequency'])
                        st.dataframe(common_df, use_container_width=True)
                    
                    with col2:
                        st.subheader("Frequency Distribution")
                        chart = create_word_frequency_chart(common_words[:10])
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
                else:
                    st.info("No common words found between the files.")
            
            if show_word_cloud:
                st.markdown("---")
                st.header("☁️ Word Clouds")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("File 1 Word Cloud")
                    if processed_text1:
                        wordcloud1 = WordCloud(width=400, height=300, background_color='white').generate(processed_text1)
                        fig1, ax1 = plt.subplots(figsize=(8, 6))
                        ax1.imshow(wordcloud1, interpolation='bilinear')
                        ax1.axis('off')
                        st.pyplot(fig1)
                
                with col2:
                    st.subheader("File 2 Word Cloud")
                    if processed_text2:
                        wordcloud2 = WordCloud(width=400, height=300, background_color='white').generate(processed_text2)
                        fig2, ax2 = plt.subplots(figsize=(8, 6))
                        ax2.imshow(wordcloud2, interpolation='bilinear')
                        ax2.axis('off')
                        st.pyplot(fig2)
            
            # Text preview
            with st.expander("📖 Text Preview"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("File 1 Preview")
                    st.text_area("", value=text1[:1000] + "..." if len(text1) > 1000 else text1, height=200, disabled=True)
                
                with col2:
                    st.subheader("File 2 Preview")
                    st.text_area("", value=text2[:1000] + "..." if len(text2) > 1000 else text2, height=200, disabled=True)
        
        else:
            st.error("Failed to extract text from one or both files. Please check the file formats and try again.")
    
    else:
        # Instructions when no files are uploaded
        st.markdown("---")
        st.info("👆 Please upload two files using the sidebar to start the similarity analysis.")
        
        # Feature explanation
        with st.expander("ℹ️ How it works"):
            st.markdown("""
            ### Cosine Similarity Analysis
            
            This tool uses **cosine similarity** with **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization to compare the content of two files.
            
            #### Process:
            1. **Text Extraction**: Extract text content from various file formats
            2. **Preprocessing**: Clean text by removing special characters, converting to lowercase, and filtering short words
            3. **Vectorization**: Convert text to numerical vectors using TF-IDF
            4. **Similarity Calculation**: Compute cosine similarity between the vectors
            
            #### Similarity Score Interpretation:
            - **0.8 - 1.0**: Very High Similarity (Nearly identical content)
            - **0.6 - 0.8**: High Similarity (Very similar content)
            - **0.4 - 0.6**: Medium Similarity (Moderately similar content)
            - **0.2 - 0.4**: Low-Medium Similarity (Some similar content)
            - **0.0 - 0.2**: Low Similarity (Very different content)
            
            #### Supported File Formats:
            - **Text files** (.txt)
            - **PDF documents** (.pdf)
            - **Word documents** (.docx, .doc)
            - **CSV files** (.csv)
            """)

if __name__ == "__main__":
    main()