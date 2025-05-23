import streamlit as st
import re
from collections import Counter
import pandas as pd

# Configure Streamlit page
st.set_page_config(
    page_title="Named Entity Recognition with ETA Algorithm",
    page_icon="🔍",
    layout="wide"
)

class ETAAlgorithm:
    """
    Entity Type Assignment (ETA) Algorithm for Named Entity Recognition
    This algorithm assigns entity types based on contextual patterns and linguistic features
    """
    
    def __init__(self):
        # Define patterns for different entity types
        self.entity_patterns = {
            'PERSON': [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last name pattern
                r'\b(?:Mr|Mrs|Ms|Dr|Prof)\. [A-Z][a-z]+\b',  # Title + name
                r'\b[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+\b',  # Three names
            ],
            'ORG': [
                r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Corporation)\b',
                r'\b(?:University|College|School) of [A-Z][a-z]+\b',
                r'\b[A-Z][a-z]+ (?:University|College|School)\b',
                r'\b(?:Apple|Microsoft|Google|Amazon|Facebook|Tesla|IBM)\b',
            ],
            'GPE': [
                r'\b[A-Z][a-z]+ (?:City|State|Country|Province)\b',
                r'\b(?:New York|Los Angeles|San Francisco|Seattle|Boston|Chicago|Miami)\b',
                r'\b(?:California|Texas|Florida|New York|Washington)\b',
                r'\b(?:USA|America|Canada|Mexico|England|France|Germany|Japan)\b',
            ],
            'MONEY': [
                r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
                r'\b\d+(?:,\d{3})* (?:dollars|euros|pounds|USD|EUR|GBP)\b',
                r'\b\$\d+(?:\.\d+)?[KMB]?\b',
            ],
            'DATE': [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b',
                r'\b\d{4}-\d{2}-\d{2}\b',
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2}, \d{4}\b',
            ],
            'TIME': [
                r'\b\d{1,2}:\d{2}(?::\d{2})? ?(?:AM|PM|am|pm)\b',
                r'\b\d{1,2}:\d{2}\b',
            ],
            'PERCENT': [
                r'\b\d+(?:\.\d+)?%\b',
                r'\b\d+(?:\.\d+)? percent\b',
            ]
        }
        
        # Context keywords for better detection
        self.context_keywords = {
            'PERSON': ['said', 'told', 'according', 'mr', 'mrs', 'dr', 'prof', 'ceo', 'president'],
            'ORG': ['company', 'corporation', 'organization', 'firm', 'business', 'enterprise'],
            'GPE': ['in', 'at', 'from', 'city', 'country', 'state', 'located', 'based'],
            'MONEY': ['cost', 'price', 'worth', 'paid', 'earned', 'revenue', 'profit', 'loss'],
            'DATE': ['on', 'during', 'since', 'until', 'by', 'when', 'date'],
            'TIME': ['at', 'during', 'until', 'by', 'when', 'time'],
            'PERCENT': ['rate', 'percentage', 'ratio', 'proportion']
        }
    
    def extract_entities_with_eta(self, text: str, selected_tags: list) -> list:
        """
        Extract entities using ETA algorithm with selected entity types
        """
        entities = []
        
        # Apply pattern matching for selected tags
        for tag in selected_tags:
            if tag in self.entity_patterns:
                for pattern in self.entity_patterns[tag]:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        entity_text = match.group()
                        start_pos = match.start()
                        end_pos = match.end()
                        
                        # Calculate confidence using ETA algorithm
                        confidence = self._calculate_eta_confidence(
                            entity_text, tag, text, start_pos, end_pos
                        )
                        
                        entities.append({
                            'text': entity_text,
                            'label': tag,
                            'start': start_pos,
                            'end': end_pos,
                            'confidence': confidence,
                            'algorithm': 'ETA'
                        })
        
        # Remove duplicates and sort by confidence
        entities = self._remove_duplicates(entities)
        entities.sort(key=lambda x: x['confidence'], reverse=True)
        
        return entities
    
    def _calculate_eta_confidence(self, entity_text: str, label: str, 
                                 full_text: str, start: int, end: int) -> float:
        """
        Calculate confidence score using ETA algorithm components
        """
        # Pattern matching confidence (40%)
        pattern_conf = self._pattern_confidence(entity_text, label)
        
        # Context similarity confidence (30%)
        context_conf = self._context_confidence(full_text, start, end, label)
        
        # Linguistic features confidence (30%)
        linguistic_conf = self._linguistic_confidence(entity_text, label)
        
        # Weighted combination
        total_confidence = (
            pattern_conf * 0.4 +
            context_conf * 0.3 +
            linguistic_conf * 0.3
        )
        
        return min(total_confidence, 1.0)
    
    def _pattern_confidence(self, entity_text: str, label: str) -> float:
        """Calculate confidence based on pattern matching"""
        if label not in self.entity_patterns:
            return 0.5
        
        for pattern in self.entity_patterns[label]:
            if re.match(pattern, entity_text, re.IGNORECASE):
                return 0.9
        
        return 0.3
    
    def _context_confidence(self, text: str, start: int, end: int, label: str) -> float:
        """Calculate confidence based on surrounding context"""
        # Get context window (30 characters before and after)
        context_start = max(0, start - 30)
        context_end = min(len(text), end + 30)
        context = text[context_start:context_end].lower()
        
        if label in self.context_keywords:
            keyword_count = sum(1 for keyword in self.context_keywords[label] 
                              if keyword in context)
            return min(keyword_count * 0.15 + 0.4, 1.0)
        
        return 0.5
    
    def _linguistic_confidence(self, entity_text: str, label: str) -> float:
        """Calculate confidence based on linguistic features"""
        # Check capitalization patterns
        if label in ['PERSON', 'ORG', 'GPE']:
            if entity_text[0].isupper():
                return 0.8
            return 0.4
        
        # Check numeric patterns
        if label in ['MONEY', 'DATE', 'TIME', 'PERCENT']:
            if any(char.isdigit() for char in entity_text):
                return 0.8
            return 0.3
        
        return 0.5
    
    def _remove_duplicates(self, entities: list) -> list:
        """Remove duplicate entities based on text and position"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity['text'], entity['start'], entity['end'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities

def visualize_entities(text: str, entities: list) -> str:
    """Create HTML visualization of entities in text"""
    if not entities:
        return text
    
    # Sort entities by start position
    sorted_entities = sorted(entities, key=lambda x: x['start'])
    
    # Color mapping for entity types
    colors = {
        'PERSON': '#FF6B6B',
        'ORG': '#4ECDC4',
        'GPE': '#45B7D1',
        'MONEY': '#96CEB4',
        'DATE': '#FFEAA7',
        'TIME': '#DDA0DD',
        'PERCENT': '#98D8C8',
    }
    
    html_text = ""
    last_end = 0
    
    for entity in sorted_entities:
        # Add text before entity
        html_text += text[last_end:entity['start']]
        
        # Add highlighted entity
        color = colors.get(entity['label'], '#CCCCCC')
        html_text += f"""
        <span style="background-color: {color}; padding: 2px 4px; margin: 1px; 
                     border-radius: 3px; font-weight: bold; color: #333;">
            {entity['text']}
            <sub style="font-size: 10px; color: #666;">
                {entity['label']} ({entity['confidence']:.2f})
            </sub>
        </span>
        """
        
        last_end = entity['end']
    
    # Add remaining text
    html_text += text[last_end:]
    
    return html_text

def main():
    st.title("🔍 Named Entity Recognition with ETA Algorithm")
    st.markdown("**Entity Type Assignment (ETA) Algorithm** - Advanced pattern-based NER system")
    st.markdown("---")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Entity type selection
    st.sidebar.subheader("Select Entity Types:")
    available_tags = ['PERSON', 'ORG', 'GPE', 'MONEY', 'DATE', 'TIME', 'PERCENT']
    
    selected_tags = []
    for tag in available_tags:
        if st.sidebar.checkbox(tag, value=True):
            selected_tags.append(tag)
    
    # ETA Algorithm explanation
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 ETA Algorithm")
    st.sidebar.info("""
    **Components:**
    - **Pattern Matching** (40%): Regex patterns for entity structures
    - **Context Analysis** (30%): Surrounding word analysis
    - **Linguistic Features** (30%): Capitalization, numeric patterns
    
    **Confidence Score**: Weighted combination of all components
    """)
    
    if not selected_tags:
        st.warning("⚠️ Please select at least one entity type from the sidebar.")
        return
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📝 Enter Text for Analysis")
        
        # Sample texts
        sample_texts = {
            "Business News": "Apple Inc. reported quarterly earnings of $25.5 billion on January 15, 2024. CEO Tim Cook will speak at the conference in San Francisco next week at 2:30 PM. The company's stock rose by 5.2% after the announcement.",
            "Personal Story": "John Smith works at Microsoft Corporation in Seattle. He earned $120,000 last year and plans to visit New York City in March 2024. His meeting is scheduled for 9:00 AM on March 15th.",
            "Academic Text": "Dr. Sarah Johnson from Harvard University published research on climate change. The study cost $2.3 million and was conducted from January 2020 to December 2023. The research shows a 15% increase in global temperatures.",
            "Financial Report": "Tesla Inc. announced record profits of $3.2 billion for Q4 2023. The electric vehicle manufacturer, based in Austin, Texas, saw a 25% increase in sales. Elon Musk will present the results on February 1, 2024 at 4:00 PM."
        }
        
        selected_sample = st.selectbox("Choose a sample text (optional):", ["Custom"] + list(sample_texts.keys()))
        
        if selected_sample != "Custom":
            default_text = sample_texts[selected_sample]
        else:
            default_text = ""
        
        user_text = st.text_area(
            "Text to analyze:",
            value=default_text,
            height=150,
            placeholder="Enter your text here..."
        )
    
    with col2:
        st.subheader("📊 Entity Types")
        for tag in available_tags:
            if tag in selected_tags:
                st.success(f"✅ {tag}")
            else:
                st.info(f"⭕ {tag}")
    
    if st.button("🚀 Analyze Text", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("⚠️ Please enter some text to analyze.")
            return
        
        # Initialize ETA algorithm
        eta_algorithm = ETAAlgorithm()
        
        # Extract entities
        with st.spinner("🔍 Analyzing text with ETA algorithm..."):
            entities = eta_algorithm.extract_entities_with_eta(user_text, selected_tags)
        
        if entities:
            # Display results
            st.success(f"✅ Found {len(entities)} entities!")
            
            # Visualized text
            st.subheader("📝 Annotated Text")
            html_text = visualize_entities(user_text, entities)
            st.markdown(html_text, unsafe_allow_html=True)
            
            # Entity details table
            st.subheader("📋 Entity Details")
            entity_df = pd.DataFrame(entities)
            entity_df = entity_df[['text', 'label', 'confidence', 'start', 'end']]
            entity_df.columns = ['Entity Text', 'Type', 'Confidence', 'Start Pos', 'End Pos']
            entity_df['Confidence'] = entity_df['Confidence'].round(3)
            st.dataframe(entity_df, use_container_width=True)
            
            # Statistics
            st.subheader("📈 Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Entities", len(entities))
            
            with col2:
                avg_confidence = sum(e['confidence'] for e in entities) / len(entities)
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
            
            with col3:
                entity_types = len(set(e['label'] for e in entities))
                st.metric("Entity Types", entity_types)
            
            # Entity type distribution
            entity_counts = Counter([e['label'] for e in entities])
            if entity_counts:
                st.subheader("🎯 Entity Distribution")
                chart_data = pd.DataFrame(list(entity_counts.items()), columns=['Entity Type', 'Count'])
                st.bar_chart(chart_data.set_index('Entity Type'))
        
        else:
            st.info("ℹ️ No entities found with the selected entity types. Try selecting different entity types or using different text.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 14px;'>
        <p>🔍 Named Entity Recognition App with ETA Algorithm | Built with Streamlit</p>
        <p>ETA Algorithm combines pattern matching, context analysis, and linguistic features for accurate entity detection</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()