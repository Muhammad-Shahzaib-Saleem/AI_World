# Named Entity Recognition with ETA Algorithm

A Streamlit web application that performs Named Entity Recognition (NER) using a custom **Entity Type Assignment (ETA) Algorithm**.

## 🚀 Features

- **Interactive Web Interface**: User-friendly Streamlit interface
- **Custom ETA Algorithm**: Advanced pattern-based entity recognition
- **Multiple Entity Types**: Support for PERSON, ORG, GPE, MONEY, DATE, TIME, PERCENT
- **Confidence Scoring**: Each entity comes with a confidence score
- **Visual Annotations**: Highlighted entities in the original text
- **Real-time Analysis**: Instant results as you type
- **Sample Texts**: Pre-loaded examples for testing

## 🧠 ETA Algorithm

The **Entity Type Assignment (ETA) Algorithm** combines three key components:

### 1. Pattern Matching (40% weight)
- Uses regular expressions to identify entity structures
- Specific patterns for each entity type (names, organizations, dates, etc.)

### 2. Context Analysis (30% weight)
- Analyzes surrounding words for contextual clues
- Uses keyword dictionaries for each entity type

### 3. Linguistic Features (30% weight)
- Considers capitalization patterns
- Analyzes numeric patterns and special characters

### Confidence Calculation
```
Confidence = (Pattern_Score × 0.4) + (Context_Score × 0.3) + (Linguistic_Score × 0.3)
```

## 📋 Supported Entity Types

| Entity Type | Description | Examples |
|-------------|-------------|----------|
| **PERSON** | People's names | John Smith, Dr. Johnson |
| **ORG** | Organizations | Apple Inc., Harvard University |
| **GPE** | Geopolitical entities | New York, California, USA |
| **MONEY** | Monetary values | $25.5 billion, 120,000 dollars |
| **DATE** | Dates | January 15, 2024, 2023-12-01 |
| **TIME** | Time expressions | 2:30 PM, 9:00 AM |
| **PERCENT** | Percentages | 5.2%, 15 percent |

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd AI_World
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run simple_ner_app.py
   ```

## 📖 Usage

1. **Select Entity Types**: Choose which types of entities you want to detect from the sidebar
2. **Enter Text**: Type your text or select a sample text
3. **Analyze**: Click the "Analyze Text" button
4. **View Results**: See highlighted entities, confidence scores, and statistics

## 🎯 Example Usage

### Input Text:
```
Apple Inc. reported quarterly earnings of $25.5 billion on January 15, 2024. 
CEO Tim Cook will speak at the conference in San Francisco next week at 2:30 PM.
```

### Expected Output:
- **Apple Inc.** (ORG, confidence: 0.85)
- **$25.5 billion** (MONEY, confidence: 0.92)
- **January 15, 2024** (DATE, confidence: 0.88)
- **Tim Cook** (PERSON, confidence: 0.78)
- **San Francisco** (GPE, confidence: 0.82)
- **2:30 PM** (TIME, confidence: 0.90)

## 🔧 Technical Details

### Algorithm Components

1. **Pattern Recognition**:
   - Regex patterns for each entity type
   - Multiple patterns per type for better coverage

2. **Context Awareness**:
   - Analyzes 30 characters before and after each entity
   - Uses contextual keywords for validation

3. **Linguistic Analysis**:
   - Capitalization patterns
   - Numeric content analysis
   - Special character recognition

### Performance Features

- **Duplicate Removal**: Eliminates overlapping entities
- **Confidence Ranking**: Sorts results by confidence score
- **Real-time Processing**: Fast analysis for interactive use

## 📊 Application Interface

### Main Features:
- **Text Input Area**: Large text box for input
- **Entity Type Selection**: Checkboxes for entity types
- **Sample Text Selector**: Pre-loaded examples
- **Results Visualization**: Color-coded entity highlighting
- **Statistics Dashboard**: Entity counts and confidence metrics
- **Detailed Table**: Complete entity information

### Visual Elements:
- Color-coded entity types
- Confidence scores displayed as subscripts
- Interactive charts and metrics
- Responsive design

## 🎨 Customization

You can easily extend the ETA algorithm by:

1. **Adding New Entity Types**:
   ```python
   self.entity_patterns['NEW_TYPE'] = [
       r'your_regex_pattern_here'
   ]
   ```

2. **Modifying Confidence Weights**:
   ```python
   pattern_conf * 0.4 +  # Adjust these weights
   context_conf * 0.3 +
   linguistic_conf * 0.3
   ```

3. **Adding Context Keywords**:
   ```python
   self.context_keywords['NEW_TYPE'] = ['keyword1', 'keyword2']
   ```

## 🚀 Deployment

The app is configured to run on port 12000 and can be accessed at:
- Local: `http://localhost:12000`
- Network: `http://0.0.0.0:12000`

For production deployment, consider using:
- Docker containers
- Cloud platforms (Heroku, AWS, GCP)
- Streamlit Cloud

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For questions or issues, please open an issue in the repository.