import streamlit as st
import nltk
import re
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge_score import rouge_scorer

# --- 1. SETUP & RESOURCE DOWNLOADING ---
@st.cache_resource
def download_nltk_resources():
    """Downloads NLTK resources once and caches them for the session."""
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

download_nltk_resources()

# --- Initialize Session State ---
# This allows Streamlit to remember data even when the page reruns
if 'generated_summary' not in st.session_state:
    st.session_state.generated_summary = None
if 'summary_metrics' not in st.session_state:
    st.session_state.summary_metrics = {}
if 'human_scores' not in st.session_state:
    st.session_state.human_scores = None

# --- 2. BACKEND LOGIC ---
def preprocess_text(sentence):
    # Case Folding & Punctuation Removal
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    
    # Tokenization
    tokens = word_tokenize(sentence)
    
    # Stopword Removal
    stop_words = set(stopwords.words('english'))
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]
    
    return " ".join(cleaned_tokens), len(cleaned_tokens)

def generate_summary(text, compression_rate=0.3):
    sentences = sent_tokenize(text)
    if not sentences:
        return "", 0, 0
        
    preprocessed_sentences = []
    word_counts = []
    
    for sentence in sentences:
        cleaned_text, count = preprocess_text(sentence)
        preprocessed_sentences.append(cleaned_text)
        word_counts.append(count)
        
    if not any(preprocessed_sentences):
        return "", len(sentences), 0

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
    
    sentence_scores = []
    for i, row in enumerate(tfidf_matrix):
        cumulative_score = np.sum(row.data)
        normalized_score = 0
        if word_counts[i] > 0:
            normalized_score = cumulative_score / word_counts[i]
        sentence_scores.append((normalized_score, i, sentences[i]))
        
    sentence_scores.sort(key=lambda x: x[0], reverse=True)
    
    num_sentences = max(1, int(len(sentences) * compression_rate))
    top_sentences = sentence_scores[:num_sentences]
    top_sentences.sort(key=lambda x: x[1])
    
    final_summary = " ".join([sent[2] for sent in top_sentences])
    
    return final_summary, len(sentences), num_sentences

def calculate_rouge(generated_summary, reference_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    return {
        "ROUGE-1": round(scores['rouge1'].fmeasure, 4),
        "ROUGE-2": round(scores['rouge2'].fmeasure, 4),
        "ROUGE-L": round(scores['rougeL'].fmeasure, 4)
    }

def calculate_compression_ratio(original_text, generated_summary):
    orig_words = len(original_text.split())
    sum_words = len(generated_summary.split())
    if orig_words == 0: return 0
    return round((sum_words / orig_words) * 100, 2)

# --- 3. FRONTEND UI (STREAMLIT) ---
st.set_page_config(page_title="TF-IDF Summarizer", page_icon="📝", layout="wide")

st.title("📝 TF-IDF Extractive Text Summarizer")
st.markdown("**Group 4 LF01** | *Efficient Information Retrieval*")
st.divider()

col_input, col_settings = st.columns([2, 1])

with col_input:
    st.subheader("1. Text Inputs")
    input_text = st.text_area(
        "Paste your Source Document here (BBC News Dataset or other):",
        height=200,
        placeholder="The sheer volume of text is constantly growing..."
    )
    
    reference_text = st.text_area(
        "Paste the Reference Summary here (Optional, required for ROUGE scores):",
        height=100,
        placeholder="Enter a standard summary to compare against our summary..."
    )

with col_settings:
    st.subheader("2. Summarization Settings")
    compression = st.slider(
        "Select Compression Rate",
        min_value=0.1, 
        max_value=0.9, 
        value=0.3, 
        step=0.05,
        help="0.3 means the summary will contain ~30% of the original sentences."
    )
    
    generate_btn = st.button("Generate Summary", type="primary", use_container_width=True)

# Generate Logic (Updates Session State)
if generate_btn:
    if not input_text.strip():
        st.warning("Please enter a source document to summarize!")
    else:
        with st.spinner("Executing TF-IDF Pipeline..."):
            summary, orig_len, sum_len = generate_summary(input_text, compression_rate=compression)
            
            # Save results to session state
            st.session_state.generated_summary = summary
            st.session_state.summary_metrics = {
                "orig_len": orig_len,
                "sum_len": sum_len,
                "comp_ratio": calculate_compression_ratio(input_text, summary),
                "rouge": calculate_rouge(summary, reference_text) if reference_text.strip() else None
            }
            # Reset human scores for the new summary
            st.session_state.human_scores = None

# Display Logic (Reads from Session State so it persists across reruns)
if st.session_state.generated_summary is not None:
    st.divider()
    st.subheader("3. Generated Summary")
    st.info(st.session_state.generated_summary)
    
    st.divider()
    st.subheader("4. Evaluation Metrics")
    
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        st.markdown("### 📈 Quantitative Assessment")
        
        # Compression Ratio
        metrics = st.session_state.summary_metrics
        st.metric(label="Compression Ratio (Word Count)", value=f"{metrics['comp_ratio']}%")
        st.caption(f"Original Sentences: {metrics['orig_len']} | Summary Sentences: {metrics['sum_len']}")
        
        # ROUGE Scores
        if metrics["rouge"]:
            st.markdown("**ROUGE Scores (F-Measure)**")
            r1, r2, rl = st.columns(3)
            r1.metric("ROUGE-1 (Unigram)", metrics["rouge"]["ROUGE-1"])
            r2.metric("ROUGE-2 (Bigram)", metrics["rouge"]["ROUGE-2"])
            rl.metric("ROUGE-L (LCS)", metrics["rouge"]["ROUGE-L"])
        else:
            st.warning("⚠️ ROUGE scores skipped. Please provide a reference summary above to calculate overlap.")
            
        # Display Human Scores if they have been submitted
        if st.session_state.human_scores:
            st.markdown("---")
            st.markdown("### 🧑‍⚖️ Human Evaluation Results")
            h1, h2, h3 = st.columns(3)
            h1.metric("Relevance", f"{st.session_state.human_scores['relevance']}/5")
            h2.metric("Coherence", f"{st.session_state.human_scores['coherence']}/5")
            h3.metric("Readability", f"{st.session_state.human_scores['readability']}/5")
            st.success("Scores successfully recorded!")
    
    with metrics_col2:
        st.markdown("### 🧑‍⚖️ Human Evaluation")
        st.write("Rate the qualitative aspects of this summary:")
        
        rel_score = st.slider("Relevance (Correlates to significant topics)", 1, 5, 3, key="rel_slider")
        coh_score = st.slider("Coherence (Transition between sentences)", 1, 5, 3, key="coh_slider")
        read_score = st.slider("Readability (Clarity, grammar, flow)", 1, 5, 3, key="read_slider")
        
        if st.button("Submit Human Evaluation"):
            # Save the human evaluation scores to session state
            st.session_state.human_scores = {
                "relevance": rel_score,
                "coherence": coh_score,
                "readability": read_score
            }
            # Force a rerun so the scores immediately render in metrics_col1
            st.rerun()
