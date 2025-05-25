import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

def calculate_cosine_similarity(text1, text2):
    """Calculate cosine similarity between two texts using TF-IDF vectors."""
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform the texts
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    return similarity

def read_file_content(uploaded_file):
    """Read content from uploaded file."""
    try:
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        # Try to read as text
        content = uploaded_file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        return content
    except UnicodeDecodeError:
        st.error("Error: Unable to decode file. Please ensure the file contains text content.")
        return None
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="File Similarity Checker",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 File Similarity Checker")
    st.markdown("Upload two text files to check their content similarity using cosine similarity.")
    
    # Create two columns for file uploads
    col1, col2 = st.columns(2)
    
    # Initialize content variables
    content1 = None
    content2 = None
    
    with col1:
        st.subheader("File 1")
        file1 = st.file_uploader(
            "Choose first file",
            type=['txt', 'md', 'py', 'js', 'html', 'css', 'json', 'csv'],
            key="file1"
        )
        
        if file1:
            st.success(f"✅ File uploaded: {file1.name}")
            content1 = read_file_content(file1)
            if content1:
                with st.expander("Preview File 1 Content"):
                    st.text_area("Content:", content1, height=200, disabled=True, key="preview1")
    
    with col2:
        st.subheader("File 2")
        file2 = st.file_uploader(
            "Choose second file",
            type=['txt', 'md', 'py', 'js', 'html', 'css', 'json', 'csv'],
            key="file2"
        )
        
        if file2:
            st.success(f"✅ File uploaded: {file2.name}")
            content2 = read_file_content(file2)
            if content2:
                with st.expander("Preview File 2 Content"):
                    st.text_area("Content:", content2, height=200, disabled=True, key="preview2")
    
    # Calculate similarity when both files are uploaded and content is available
    if file1 and file2 and content1 and content2:
        st.markdown("---")
        
        if st.button("🔍 Calculate Similarity", type="primary"):
            with st.spinner("Calculating similarity..."):
                try:
                    similarity_score = calculate_cosine_similarity(content1, content2)
                    
                    # Display results
                    st.subheader("📊 Similarity Results")
                    
                    # Create metrics display
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="Cosine Similarity",
                            value=f"{similarity_score:.4f}",
                            delta=f"{similarity_score * 100:.2f}%"
                        )
                    
                    with col2:
                        if similarity_score > 0.8:
                            similarity_level = "Very High"
                            color = "🟢"
                        elif similarity_score > 0.6:
                            similarity_level = "High"
                            color = "🟡"
                        elif similarity_score > 0.4:
                            similarity_level = "Medium"
                            color = "🟠"
                        elif similarity_score > 0.2:
                            similarity_level = "Low"
                            color = "🔴"
                        else:
                            similarity_level = "Very Low"
                            color = "⚫"
                        
                        st.metric(
                            label="Similarity Level",
                            value=f"{color} {similarity_level}"
                        )
                    
                    with col3:
                        st.metric(
                            label="Files Compared",
                            value="2",
                            delta=f"{file1.name} vs {file2.name}"
                        )
                    
                    # Progress bar visualization
                    st.subheader("📈 Similarity Visualization")
                    progress_col1, progress_col2 = st.columns([3, 1])
                    
                    with progress_col1:
                        st.progress(similarity_score)
                    
                    with progress_col2:
                        st.write(f"**{similarity_score * 100:.2f}%**")
                    
                    # Interpretation
                    st.subheader("🔍 Interpretation")
                    if similarity_score > 0.8:
                        st.success("The files are very similar in content. They likely contain very similar or nearly identical information.")
                    elif similarity_score > 0.6:
                        st.info("The files have high similarity. They share significant common content or themes.")
                    elif similarity_score > 0.4:
                        st.warning("The files have moderate similarity. They share some common elements but also have distinct differences.")
                    elif similarity_score > 0.2:
                        st.warning("The files have low similarity. They share few common elements.")
                    else:
                        st.error("The files have very low similarity. They appear to be quite different in content.")
                    
                    # Additional statistics
                    with st.expander("📋 Additional Statistics"):
                        st.write(f"**File 1 ({file1.name}):**")
                        st.write(f"- Character count: {len(content1):,}")
                        st.write(f"- Word count: {len(content1.split()):,}")
                        st.write(f"- Line count: {len(content1.splitlines()):,}")
                        
                        st.write(f"**File 2 ({file2.name}):**")
                        st.write(f"- Character count: {len(content2):,}")
                        st.write(f"- Word count: {len(content2.split()):,}")
                        st.write(f"- Line count: {len(content2.splitlines()):,}")
                        
                except Exception as e:
                    st.error(f"Error calculating similarity: {str(e)}")
    
    # Information section
    st.markdown("---")
    with st.expander("ℹ️ About Cosine Similarity"):
        st.markdown("""
        **Cosine Similarity** measures the cosine of the angle between two vectors in a multi-dimensional space.
        
        - **Range**: 0 to 1 (for text similarity)
        - **0**: Completely different (orthogonal vectors)
        - **1**: Identical content (same direction)
        
        **How it works:**
        1. Convert text to TF-IDF (Term Frequency-Inverse Document Frequency) vectors
        2. Calculate the cosine of the angle between these vectors
        3. Higher values indicate more similar content
        
        **Supported file types:** .txt, .md, .py, .js, .html, .css, .json, .csv
        """)

if __name__ == "__main__":
    main()