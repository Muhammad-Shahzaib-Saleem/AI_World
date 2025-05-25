#!/usr/bin/env python3
"""
Test script to verify the cosine similarity functionality
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(text1, text2):
    """Calculate cosine similarity between two texts using TF-IDF vectors."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity

def test_with_sample_files():
    """Test the similarity calculation with sample files."""
    
    # Read sample files
    with open('sample_files/sample1.txt', 'r') as f:
        content1 = f.read()
    
    with open('sample_files/sample2.txt', 'r') as f:
        content2 = f.read()
    
    with open('sample_files/sample3.txt', 'r') as f:
        content3 = f.read()
    
    # Calculate similarities
    sim_1_2 = calculate_cosine_similarity(content1, content2)
    sim_1_3 = calculate_cosine_similarity(content1, content3)
    sim_2_3 = calculate_cosine_similarity(content2, content3)
    
    print("File Similarity Test Results:")
    print("=" * 50)
    print(f"Sample1 vs Sample2 (AI/ML topics): {sim_1_2:.4f} ({sim_1_2*100:.2f}%)")
    print(f"Sample1 vs Sample3 (AI vs Web Dev): {sim_1_3:.4f} ({sim_1_3*100:.2f}%)")
    print(f"Sample2 vs Sample3 (AI vs Web Dev): {sim_2_3:.4f} ({sim_2_3*100:.2f}%)")
    print("=" * 50)
    
    # Test with identical content
    identical_sim = calculate_cosine_similarity(content1, content1)
    print(f"Identical content similarity: {identical_sim:.4f} ({identical_sim*100:.2f}%)")
    
    # Test with completely different content
    different_text1 = "The quick brown fox jumps over the lazy dog."
    different_text2 = "Python programming language machine learning algorithms."
    different_sim = calculate_cosine_similarity(different_text1, different_text2)
    print(f"Different content similarity: {different_sim:.4f} ({different_sim*100:.2f}%)")

if __name__ == "__main__":
    test_with_sample_files()