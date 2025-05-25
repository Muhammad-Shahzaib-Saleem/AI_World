import streamlit as st
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import arabic_reshaper
from bidi.algorithm import get_display
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="📖 Holy Quran Word Cloud Generator",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E8B57, #3CB371, #90EE90);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E8B57;
        margin: 0.5rem 0;
    }
    
    .stSelectbox > div > div {
        background-color: #f0f8f0;
    }
    
    .stSlider > div > div {
        background-color: #f0f8f0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f0f8f0 0%, #e8f5e8 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_quran_data():
    """Load and cache the Quran dataset"""
    try:
        # Try to load from uploaded file first
        if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
            df = pd.read_csv(st.session_state.uploaded_file)
        else:
            # Use sample data if no file uploaded
            df = create_sample_quran_data()
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return create_sample_quran_data()

def create_sample_quran_data():
    """Create sample Quran data for demonstration"""
    sample_data = {
        'surah_no': [1, 1, 1, 1, 1, 1, 1, 2, 2, 2] * 10,
        'ayah_no': list(range(1, 8)) + [1, 2, 3] + list(range(1, 8)) * 9 + [1, 2, 3] * 9,
        'surah_name': ['Al-Fatihah'] * 7 + ['Al-Baqarah'] * 3 + ['Al-Fatihah'] * 63 + ['Al-Baqarah'] * 27,
        'ayah_text': [
            'In the name of Allah, the Entirely Merciful, the Especially Merciful',
            'All praise is due to Allah, Lord of the worlds',
            'The Entirely Merciful, the Especially Merciful',
            'Sovereign of the Day of Recompense',
            'It is You we worship and You we ask for help',
            'Guide us to the straight path',
            'The path of those upon whom You have bestowed favor, not of those who have evoked Your anger or of those who are astray',
            'This is the Book about which there is no doubt, a guidance for those conscious of Allah',
            'Who believe in the unseen, establish prayer, and spend out of what We have provided for them',
            'And who believe in what has been revealed to you, and what was revealed before you, and of the Hereafter they are certain'
        ] * 10
    }
    return pd.DataFrame(sample_data)

def preprocess_text(text, language='english'):
    """Preprocess text for word cloud generation"""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def get_stopwords(language='english'):
    """Get stopwords for different languages"""
    english_stopwords = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'a', 'an', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
        'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
        'its', 'our', 'their', 'who', 'what', 'when', 'where', 'why', 'how'
    }
    
    arabic_stopwords = {
        'في', 'من', 'إلى', 'على', 'عن', 'مع', 'بعد', 'قبل', 'عند', 'لدى',
        'هذا', 'هذه', 'ذلك', 'تلك', 'التي', 'الذي', 'التي', 'اللذان', 'اللتان',
        'الذين', 'اللواتي', 'اللاتي', 'ما', 'من', 'أن', 'إن', 'كان', 'كانت'
    }
    
    if language == 'arabic':
        return arabic_stopwords
    else:
        return english_stopwords

def generate_wordcloud(text, width=800, height=400, max_words=100, colormap='viridis', background_color='white'):
    """Generate word cloud from text"""
    if not text or len(text.strip()) == 0:
        return None
    
    stopwords = get_stopwords('english')
    
    wordcloud = WordCloud(
        width=width,
        height=height,
        max_words=max_words,
        colormap=colormap,
        background_color=background_color,
        stopwords=stopwords,
        relative_scaling=0.5,
        random_state=42
    ).generate(text)
    
    return wordcloud

def create_frequency_chart(text, top_n=20):
    """Create frequency chart of most common words"""
    if not text:
        return None
    
    words = text.split()
    stopwords = get_stopwords('english')
    filtered_words = [word for word in words if word.lower() not in stopwords and len(word) > 2]
    
    word_freq = Counter(filtered_words)
    top_words = word_freq.most_common(top_n)
    
    if not top_words:
        return None
    
    words, frequencies = zip(*top_words)
    
    fig = px.bar(
        x=list(frequencies),
        y=list(words),
        orientation='h',
        title=f'Top {top_n} Most Frequent Words',
        labels={'x': 'Frequency', 'y': 'Words'},
        color=list(frequencies),
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    
    return fig

def create_surah_analysis(df):
    """Create analysis by Surah"""
    if df.empty:
        return None, None
    
    # Surah statistics
    surah_stats = df.groupby('surah_name').agg({
        'ayah_no': 'count',
        'ayah_text': lambda x: ' '.join(x.astype(str))
    }).reset_index()
    
    surah_stats.columns = ['Surah', 'Ayah_Count', 'Full_Text']
    surah_stats['Word_Count'] = surah_stats['Full_Text'].apply(lambda x: len(str(x).split()))
    surah_stats['Avg_Words_Per_Ayah'] = surah_stats['Word_Count'] / surah_stats['Ayah_Count']
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Ayah Count by Surah', 'Word Count by Surah', 
                       'Average Words per Ayah', 'Surah Length Distribution'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "histogram"}]]
    )
    
    # Ayah count
    fig.add_trace(
        go.Bar(x=surah_stats['Surah'], y=surah_stats['Ayah_Count'], 
               name='Ayah Count', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Word count
    fig.add_trace(
        go.Bar(x=surah_stats['Surah'], y=surah_stats['Word_Count'], 
               name='Word Count', marker_color='lightgreen'),
        row=1, col=2
    )
    
    # Average words per ayah
    fig.add_trace(
        go.Bar(x=surah_stats['Surah'], y=surah_stats['Avg_Words_Per_Ayah'], 
               name='Avg Words/Ayah', marker_color='lightcoral'),
        row=2, col=1
    )
    
    # Distribution
    fig.add_trace(
        go.Histogram(x=surah_stats['Word_Count'], name='Distribution', 
                    marker_color='lightyellow'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False, title_text="Surah Analysis Dashboard")
    
    return fig, surah_stats

def main():
    # Header
    st.markdown('<h1 class="main-header">📖 Holy Quran Word Cloud Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Visualize the beautiful words of the Holy Quran through interactive word clouds and analytics</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Quran Dataset (CSV)",
        type=['csv'],
        help="Upload a CSV file with columns: surah_no, ayah_no, surah_name, ayah_text"
    )
    
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
    
    # Load data
    df = load_quran_data()
    
    if df.empty:
        st.error("No data available. Please upload a valid CSV file.")
        return
    
    # Sidebar controls
    st.sidebar.markdown("## ⚙️ Word Cloud Settings")
    
    # Surah selection
    available_surahs = ['All Surahs'] + sorted(df['surah_name'].unique().tolist())
    selected_surah = st.sidebar.selectbox("Select Surah", available_surahs)
    
    # Word cloud parameters
    width = st.sidebar.slider("Width", 400, 1200, 800)
    height = st.sidebar.slider("Height", 300, 800, 400)
    max_words = st.sidebar.slider("Maximum Words", 50, 200, 100)
    
    colormap_options = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greens', 'Blues', 'Reds']
    colormap = st.sidebar.selectbox("Color Scheme", colormap_options)
    
    background_color = st.sidebar.color_picker("Background Color", "#FFFFFF")
    
    # Analysis options
    st.sidebar.markdown("## 📊 Analysis Options")
    show_frequency = st.sidebar.checkbox("Show Word Frequency Chart", True)
    show_surah_analysis = st.sidebar.checkbox("Show Surah Analysis", True)
    top_words_count = st.sidebar.slider("Top Words to Display", 10, 50, 20)
    
    # Filter data based on selection
    if selected_surah != 'All Surahs':
        filtered_df = df[df['surah_name'] == selected_surah]
        st.info(f"📖 Analyzing: **{selected_surah}** ({len(filtered_df)} ayahs)")
    else:
        filtered_df = df
        st.info(f"📖 Analyzing: **All Surahs** ({len(filtered_df)} ayahs)")
    
    # Prepare text
    all_text = ' '.join(filtered_df['ayah_text'].astype(str))
    processed_text = preprocess_text(all_text)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎨 Word Cloud", "📊 Frequency Analysis", "📈 Surah Analytics", "📋 Data Explorer"])
    
    with tab1:
        st.markdown("### 🎨 Word Cloud Visualization")
        
        if processed_text:
            # Generate word cloud
            wordcloud = generate_wordcloud(
                processed_text, width, height, max_words, colormap, background_color
            )
            
            if wordcloud:
                # Display word cloud
                fig, ax = plt.subplots(figsize=(width/100, height/100))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                
                # Download button
                img_buffer = io.BytesIO()
                wordcloud.to_image().save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download Word Cloud",
                    data=img_buffer.getvalue(),
                    file_name=f"quran_wordcloud_{selected_surah.replace(' ', '_')}.png",
                    mime="image/png"
                )
            else:
                st.warning("Unable to generate word cloud. Please check your data.")
        else:
            st.warning("No text data available for word cloud generation.")
    
    with tab2:
        st.markdown("### 📊 Word Frequency Analysis")
        
        if show_frequency and processed_text:
            freq_chart = create_frequency_chart(processed_text, top_words_count)
            if freq_chart:
                st.plotly_chart(freq_chart, use_container_width=True)
                
                # Word statistics
                words = processed_text.split()
                stopwords = get_stopwords('english')
                filtered_words = [word for word in words if word.lower() not in stopwords and len(word) > 2]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Total Words", len(words))
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Unique Words", len(set(filtered_words)))
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Avg Word Length", f"{np.mean([len(word) for word in filtered_words]):.1f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Vocabulary Richness", f"{len(set(filtered_words))/len(filtered_words)*100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Unable to generate frequency analysis.")
    
    with tab3:
        st.markdown("### 📈 Surah Analytics Dashboard")
        
        if show_surah_analysis:
            surah_fig, surah_stats = create_surah_analysis(df)
            if surah_fig:
                st.plotly_chart(surah_fig, use_container_width=True)
                
                st.markdown("#### 📋 Surah Statistics Table")
                st.dataframe(
                    surah_stats[['Surah', 'Ayah_Count', 'Word_Count', 'Avg_Words_Per_Ayah']],
                    use_container_width=True
                )
            else:
                st.warning("Unable to generate surah analysis.")
    
    with tab4:
        st.markdown("### 📋 Data Explorer")
        
        # Dataset overview
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📊 Dataset Overview")
            st.write(f"**Total Ayahs:** {len(df)}")
            st.write(f"**Total Surahs:** {df['surah_name'].nunique()}")
            st.write(f"**Dataset Shape:** {df.shape}")
        
        with col2:
            st.markdown("#### 🔍 Sample Data")
            st.dataframe(df.head(), use_container_width=True)
        
        # Full dataset
        st.markdown("#### 📖 Complete Dataset")
        st.dataframe(df, use_container_width=True)
        
        # Download filtered data
        csv_buffer = io.StringIO()
        filtered_df.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"quran_data_{selected_surah.replace(' ', '_')}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>📖 Holy Quran Word Cloud Generator | Built with ❤️ using Streamlit</p>
        <p>May Allah bless this effort and make it beneficial for all 🤲</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()