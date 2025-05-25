# 📖 Holy Quran Word Cloud Generator

A beautiful and interactive Streamlit application for generating word clouds and analyzing text patterns from the Holy Quran dataset. This application provides comprehensive text analytics, visualization tools, and insights into the sacred text.

## ✨ Features

### 🎨 Word Cloud Generation
- **Interactive Word Clouds**: Generate beautiful word clouds with customizable parameters
- **Multiple Color Schemes**: Choose from various color palettes (viridis, plasma, inferno, etc.)
- **Customizable Dimensions**: Adjust width, height, and maximum words
- **Background Color Selection**: Pick custom background colors
- **Download Functionality**: Save word clouds as PNG images

### 📊 Text Analytics
- **Word Frequency Analysis**: Interactive bar charts showing most frequent words
- **Text Statistics**: Total words, unique words, average word length, vocabulary richness
- **Stopword Filtering**: Intelligent removal of common words for better insights
- **Top N Words**: Configurable number of top words to display

### 📈 Surah Analysis
- **Surah-wise Statistics**: Analyze individual Surahs or all together
- **Comparative Analytics**: Ayah count, word count, and average words per Ayah
- **Interactive Dashboard**: Multi-panel visualization with various chart types
- **Distribution Analysis**: Statistical distribution of Surah lengths

### 📋 Data Management
- **CSV Upload**: Support for custom Quran datasets
- **Data Explorer**: Browse and examine the dataset
- **Filtered Downloads**: Export filtered data as CSV
- **Sample Data**: Built-in sample data for demonstration

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd word_cloud_holy_quran
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run app.py
```

4. **Access the app:**
Open your browser and navigate to `http://localhost:8501`

### Dataset Format

The application expects a CSV file with the following columns:
- `surah_no`: Surah number (integer)
- `ayah_no`: Ayah number within the Surah (integer)
- `surah_name`: Name of the Surah (string)
- `ayah_text`: Text content of the Ayah (string)

Example:
```csv
surah_no,ayah_no,surah_name,ayah_text
1,1,Al-Fatihah,"In the name of Allah, the Entirely Merciful, the Especially Merciful"
1,2,Al-Fatihah,"All praise is due to Allah, Lord of the worlds"
```

## 📖 Usage Guide

### 1. Data Upload
- Use the sidebar to upload your Quran dataset (CSV format)
- If no file is uploaded, the app uses built-in sample data
- Verify your data in the "Data Explorer" tab

### 2. Word Cloud Generation
- **Select Surah**: Choose a specific Surah or analyze all Surahs
- **Customize Appearance**: Adjust dimensions, colors, and word count
- **Generate**: The word cloud updates automatically
- **Download**: Save your word cloud as a PNG image

### 3. Frequency Analysis
- View interactive bar charts of most frequent words
- Examine text statistics and vocabulary metrics
- Adjust the number of top words to display

### 4. Surah Analytics
- Compare statistics across different Surahs
- Analyze word distribution patterns
- Export statistical data for further analysis

### 5. Data Exploration
- Browse the complete dataset
- View sample data and dataset overview
- Download filtered data for external analysis

## 🎯 Key Features Explained

### Word Cloud Customization
- **Width/Height**: Control the dimensions of your word cloud
- **Max Words**: Limit the number of words displayed
- **Color Schemes**: Choose from scientific color palettes
- **Background**: Select custom background colors

### Text Preprocessing
- **Stopword Removal**: Filters common words (the, and, or, etc.)
- **Case Normalization**: Converts text to lowercase
- **Special Character Removal**: Cleans punctuation and numbers
- **Whitespace Normalization**: Removes extra spaces

### Analytics Metrics
- **Total Words**: Count of all words in the text
- **Unique Words**: Count of distinct words
- **Average Word Length**: Mean character count per word
- **Vocabulary Richness**: Ratio of unique to total words

## 🛠️ Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **WordCloud**: Word cloud generation library
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **Matplotlib/Seaborn**: Static plotting libraries
- **Arabic-Reshaper/Python-BIDI**: Arabic text support

### Performance Optimizations
- **Caching**: Data loading is cached for better performance
- **Efficient Processing**: Optimized text preprocessing
- **Memory Management**: Proper handling of large datasets

## 🎨 Customization Options

### Color Schemes
- `viridis`: Blue to yellow gradient
- `plasma`: Purple to pink gradient
- `inferno`: Black to yellow gradient
- `magma`: Black to white gradient
- `cividis`: Blue to yellow (colorblind-friendly)
- `Greens`, `Blues`, `Reds`: Single-color gradients

### Layout Options
- **Wide Layout**: Full-screen utilization
- **Sidebar Controls**: Organized parameter controls
- **Tabbed Interface**: Organized content sections
- **Responsive Design**: Adapts to different screen sizes

## 📊 Sample Outputs

The application generates various types of visualizations:

1. **Word Clouds**: Beautiful visual representations of word frequency
2. **Bar Charts**: Horizontal frequency charts with color coding
3. **Statistical Dashboards**: Multi-panel analytics views
4. **Data Tables**: Interactive, sortable data displays

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Holy Quran**: The source of this sacred text
- **Streamlit Community**: For the amazing framework
- **WordCloud Library**: For word cloud generation capabilities
- **Plotly**: For interactive visualizations

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section below
2. Review the documentation
3. Submit an issue on GitHub

## 🔧 Troubleshooting

### Common Issues

**1. File Upload Problems**
- Ensure your CSV file has the required columns
- Check for proper encoding (UTF-8 recommended)
- Verify file size is reasonable

**2. Word Cloud Not Generating**
- Check if text data is available
- Verify text preprocessing is working
- Ensure sufficient unique words exist

**3. Performance Issues**
- Try reducing the dataset size
- Lower the maximum words parameter
- Use smaller word cloud dimensions

**4. Arabic Text Display**
- Ensure proper font support
- Check text encoding
- Verify Arabic reshaping libraries are installed

## 🌟 Future Enhancements

- **Multi-language Support**: Enhanced Arabic text processing
- **Advanced Analytics**: Sentiment analysis, topic modeling
- **Export Options**: PDF reports, multiple image formats
- **Comparison Tools**: Side-by-side Surah comparisons
- **Search Functionality**: Find specific words or phrases
- **Theme Customization**: Dark mode, custom color themes

---

**May Allah bless this effort and make it beneficial for all who seek to understand and appreciate the beauty of the Holy Quran. 🤲**