"""
Frontend Module: Streamlit UI for Indian Exam Recommendation System
Version: 4.0 - Integrated Machine Learning (without TensorFlow)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from chatbot import generate_personalized_guidance

import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import backend modules
try:
    from backend import (
        get_recommendation_system, 
        Stream, 
        validate_student_data
    )
    from database import get_database_manager
except ImportError as e:
    st.error(f"❌ Module import error: {e}")
    st.error("Make sure backend.py and database.py are in the same directory.")
    st.stop()

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="🇮🇳 ExamGenius AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= CSS =================
def load_custom_css():
    st.markdown("""
    <style>

    /* ---------- GLOBAL ---------- */
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont;
    }

    body {
        background: radial-gradient(circle at top, #0f2027, #020617);
        color: #e5e7eb;
    }

    /* ---------- HEADERS ---------- */
    .main-header {
        font-size: 3rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #22d3ee, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        text-align: center;
        font-size: 1.1rem;
        color: #9ca3af;
        margin-bottom: 2rem;
    }

    /* ---------- METRIC CARDS ---------- */
    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }

    /* ---------- TABS ---------- */
    button[data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
        color: #9ca3af;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #22d3ee;
        border-bottom: 3px solid #22d3ee;
    }

    /* ---------- EXAM CARD ---------- */
    .exam-card {
        background: linear-gradient(145deg, #020617, #020617);
        border-radius: 18px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 
            0 10px 30px rgba(0,0,0,0.4),
            inset 0 0 0 1px rgba(255,255,255,0.03);
        transition: all 0.25s ease;
    }

    .exam-card:hover {
        transform: translateY(-6px) scale(1.01);
        box-shadow: 0 18px 45px rgba(0,0,0,0.6);
        border-color: rgba(34,211,238,0.4);
    }

    .exam-card h4 {
        font-size: 1.25rem;
        font-weight: 800;
        color: #f9fafb;
        margin-bottom: 0.3rem;
    }

    .exam-card p {
        color: #9ca3af;
        font-size: 0.95rem;
        margin-bottom: 0.8rem;
    }

    /* ---------- BADGES ---------- */
    .feature-badge {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(34,211,238,0.15);
        color: #22d3ee;
        border: 1px solid rgba(34,211,238,0.3);
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }

    /* ---------- ML SCORE BADGE ---------- */
    .feature-badge.ml-high {
        background: rgba(16,185,129,0.2);
        color: #10b981;
        border-color: rgba(16,185,129,0.4);
    }

    .feature-badge.ml-mid {
        background: rgba(251,191,36,0.2);
        color: #fbbf24;
        border-color: rgba(251,191,36,0.4);
    }

    .feature-badge.ml-low {
        background: rgba(239,68,68,0.2);
        color: #ef4444;
        border-color: rgba(239,68,68,0.4);
    }

    /* ---------- COLUMNS TEXT ---------- */
    .exam-card strong {
        color: #e5e7eb;
        font-weight: 600;
    }

    /* ---------- SUCCESS / INFO ---------- */
    div[data-testid="stAlert"] {
        border-radius: 14px;
        font-weight: 600;
    }

    /* ---------- DATAFRAMES ---------- */
    .stDataFrame {
        background: rgba(255,255,255,0.03);
        border-radius: 14px;
        padding: 0.5rem;
    }

    /* ---------- FOOTER ---------- */
    footer {
        visibility: hidden;
    }

    </style>
    """, unsafe_allow_html=True)


# ================= ML MODELS =================

class ExamMLRecommender:
    """Machine Learning based exam recommendation system"""
    
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_training_data(self, student_data, exam_data):
        """Prepare training data for ML models"""
        features = []
        labels = []
        
        # Feature engineering
        for student in student_data:
            for exam in exam_data:
                # Student features
                student_features = [
                    self._encode_stream(student['stream']),
                    student.get('percentage', 50) / 100,  # Normalize percentage
                    student.get('age', 18) / 100,  # Normalize age
                    1 if student.get('scholarship_needed', False) else 0,
                    self._encode_class(student['current_class'])
                ]
                
                # Exam features
                exam_features = [
                    self._encode_stream(exam['stream']),
                    self._encode_level(exam['level']),
                    self._encode_class(exam['eligible_from']),
                    1 if exam.get('scholarship_available', False) else 0,
                    exam.get('min_percentage', 0) / 100 if exam.get('min_percentage') else 0
                ]
                
                # Combined features
                combined_features = student_features + exam_features
                
                # Similarity features
                similarity_features = [
                    1 if student['stream'] == exam['stream'] or exam['stream'] == 'All' else 0,
                    1 if student['state'] == exam.get('state_specific', 'All-India') else 0,
                    self._calculate_class_compatibility(student['current_class'], exam['eligible_from'])
                ]
                
                final_features = combined_features + similarity_features
                features.append(final_features)
                
                # Label (1 if exam is relevant, 0 otherwise)
                label = 1 if (
                    (student['stream'] == exam['stream'] or exam['stream'] == 'All') and
                    self._is_class_eligible(student['current_class'], exam['eligible_from'])
                ) else 0
                labels.append(label)
        
        return np.array(features), np.array(labels)
    
    def train_models(self, features, labels):
        """Train ML models"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.rf_model.fit(X_train_scaled, y_train)
        rf_pred = self.rf_model.predict(X_test_scaled)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        # Train Gradient Boosting
        self.gb_model.fit(X_train_scaled, y_train)
        gb_pred = self.gb_model.predict(X_test_scaled)
        gb_accuracy = accuracy_score(y_test, gb_pred)
        
        self.is_trained = True
        
        return {
            'rf_accuracy': rf_accuracy,
            'gb_accuracy': gb_accuracy,
            'rf_report': classification_report(y_test, rf_pred, output_dict=True),
            'gb_report': classification_report(y_test, gb_pred, output_dict=True)
        }
    
    def predict_relevance(self, student_profile, exam):
        """Predict relevance score using ML models"""
        if not self.is_trained:
            return 0.5  # Default score
        
        # Prepare features for prediction
        student_features = [
            self._encode_stream(student_profile.stream),
            (student_profile.percentage or 50) / 100,
            (student_profile.age or 18) / 100,
            1 if student_profile.scholarship_needed else 0,
            self._encode_class(student_profile.current_class)
        ]
        
        exam_features = [
            self._encode_stream(exam.stream),
            self._encode_level(exam.level),
            self._encode_class(exam.eligible_from),
            1 if exam.scholarship_available else 0,
            (exam.min_percentage or 0) / 100
        ]
        
        similarity_features = [
            1 if student_profile.stream == exam.stream or exam.stream == 'All' else 0,
            1 if student_profile.state == getattr(exam, 'state_specific', 'All-India') else 0,
            self._calculate_class_compatibility(student_profile.current_class, exam.eligible_from)
        ]
        
        features = student_features + exam_features + similarity_features
        features_scaled = self.scaler.transform([features])
        
        # Get predictions from both models
        rf_score = self.rf_model.predict_proba(features_scaled)[0][1]
        gb_score = self.gb_model.predict_proba(features_scaled)[0][1]
        
        # Ensemble average
        return (rf_score + gb_score) / 2
    
    def _encode_stream(self, stream):
        """Encode stream to numeric"""
        streams = list(Stream)
        return streams.index(Stream(stream)) if stream in [s.value for s in streams] else len(streams)
    
    def _encode_level(self, level):
        """Encode level to numeric"""
        levels = ['School', 'UG', 'PG', 'PhD', 'Diploma', 'Professional']
        return levels.index(level) if level in levels else len(levels)
    
    def _encode_class(self, class_level):
        """Encode class level to numeric"""
        classes = ['8', '9', '10', '11', '12', 'Diploma', 'UG', 'PG', 'PhD']
        return classes.index(class_level) if class_level in classes else len(classes)
    
    def _calculate_class_compatibility(self, student_class, exam_eligible_from):
        """Calculate class compatibility score"""
        class_order = ['8', '9', '10', '11', '12', 'Diploma', 'UG', 'PG', 'PhD']
        try:
            student_idx = class_order.index(student_class)
            exam_idx = class_order.index(exam_eligible_from)
            return 1.0 if student_idx >= exam_idx else 0.0
        except:
            return 0.5
    
    def _is_class_eligible(self, student_class, exam_eligible_from):
        """Check if student class is eligible for exam"""
        eligibility_map = {
            '8': ['8', '9', '10', '11', '12', 'Diploma', 'UG', 'PG', 'PhD'],
            '9': ['9', '10', '11', '12', 'Diploma', 'UG', 'PG', 'PhD'],
            '10': ['10', '11', '12', 'Diploma', 'UG', 'PG', 'PhD'],
            '11': ['11', '12', 'Diploma', 'UG', 'PG', 'PhD'],
            '12': ['12', 'Diploma', 'UG', 'PG', 'PhD'],
            'Diploma': ['Diploma', 'UG', 'PG', 'PhD'],
            'UG': ['UG', 'PG', 'PhD'],
            'PG': ['PG', 'PhD'],
            'PhD': ['PhD']
        }
        return exam_eligible_from in eligibility_map.get(student_class, [])

# ================= SESSION =================

def initialize_session_state():
    """Initialize session state variables"""
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = None
    if "student_profile" not in st.session_state:
        st.session_state.student_profile = None
    if "ml_recommender" not in st.session_state:
        st.session_state.ml_recommender = ExamMLRecommender()
    if "ml_trained" not in st.session_state:
        st.session_state.ml_trained = False
    if "university_exams" not in st.session_state:
        st.session_state.university_exams = []

# ================= INIT SYSTEM =================

def initialize_systems():
    """Initialize backend and database systems"""
    backend = get_recommendation_system()
    database = get_database_manager(use_sqlite=False)
    return backend, database

# ================= HEADER =================

def render_header():
    """Render the main header"""
    st.markdown('<h1 class="main-header">🎯 ExamGenius AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header"> This platform tell you about which Exams or which Up-Coming Exams you will be Eligible </p>', unsafe_allow_html=True)
    

# ================= ML TRAINING SECTION =================

def render_ml_training_section():
    """Render ML training interface"""
    with st.expander("🎯 AI Model", expanded=False):
                
        if st.button("🚀 AI Score", use_container_width=True):
            with st.spinner("Check Score..."):
                try:
                    # Generate synthetic training data
                    synthetic_data = generate_synthetic_training_data()
                    features, labels = st.session_state.ml_recommender.prepare_training_data(
                        synthetic_data['students'],
                        synthetic_data['exams']
                    )
                    
                    # Train models
                    results = st.session_state.ml_recommender.train_models(features, labels)
                    
                    st.session_state.ml_trained = True
                    
                    # Display results
                    st.success("✅ Result Declared Successfully!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Random Forest Accuracy", f"{results['rf_accuracy']:.2%}")
                    with col2:
                        st.metric("Gradient Boosting Accuracy", f"{results['gb_accuracy']:.2%}")
                        
                except Exception as e:
                    st.error(f"Error getting Score: {e}")
        
        
def generate_synthetic_training_data():
    """Generate synthetic training data"""
    # Synthetic student profiles
    students = []
    streams = [s.value for s in Stream]
    classes = ['8', '9', '10', '11', '12', 'UG', 'PG']
    states = ['All-India']
    
    for i in range(100):
        students.append({
            'current_class': np.random.choice(classes),
            'stream': np.random.choice(streams),
            'state': np.random.choice(states),
            'percentage': np.random.uniform(50, 100),
            'age': np.random.randint(15, 25),
            'scholarship_needed': np.random.choice([True, False])
        })
    
    # Synthetic exam data
    exams = []
    exam_categories = ['Engineering', 'Medical', 'Management', 'Law', 'Government Job', 'Banking']
    
    for i in range(50):
        exams.append({
            'stream': np.random.choice(streams + ['All']),
            'level': np.random.choice(['School', 'UG', 'PG', 'PhD']),
            'eligible_from': np.random.choice(classes),
            'scholarship_available': np.random.choice([True, False]),
            'min_percentage': np.random.uniform(40, 90),
            'state_specific': np.random.choice(states + [None])
        })
    
    return {'students': students, 'exams': exams}

# ================= STUDENT FORM =================

def render_student_form():
    """Render the student input form"""
    st.markdown("## 🎓 Find Your Perfect Exam Match")
    st.markdown("Enter your details below to get personalized exam recommendations.")
    
    with st.form("student_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Academic Level
            current_class = st.selectbox(
                "Current Class/Level *",
                options=["8", "9", "10", "11", "12", "Diploma", "UG", "PG", "PhD"],
                help="Select your current academic level"
            )
            
            # Scholarship Need
            scholarship_needed = st.selectbox(
                "Do you need Scholarships? *",
                options=["No", "Yes"],
                help="Indicate if you require financial assistance"
            )
            
        with col2:
            # Academic Stream
            stream_options = [stream.value for stream in Stream]
            stream = st.selectbox(
                "Your Academic Stream *",
                options=stream_options,
                help="Select your academic specialization"
            )
            
            # Education Board (only for school levels)
            if current_class in ["8", "9", "10", "11", "12"]:
                board = st.selectbox(
                    "Education Board *",
                    options=["CBSE", "ICSE", "State Board", "Other", "N/A"]
                )
            else:
                board = "N/A"
                st.info("Board selection not applicable for higher education levels")
            
        with col3:
            # State
            state = st.selectbox(
                "State *",
                options=["All-India", "Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", 
                        "Uttar Pradesh", "West Bengal", "Gujarat", "Rajasthan", "Kerala",
                        "Other"],
                help="Select your state for state-specific exams"
            )
            
            # Optional Fields
            with st.expander("Additional Information (Improves ML Accuracy)"):
                age = st.number_input("Age", min_value=10, max_value=50, value=None, step=1)
                percentage = st.number_input("Percentage/CGPA", min_value=0.0, max_value=100.0, 
                                           value=None, step=0.1, format="%.1f")
        
        # Submit Button
        submit_col1, submit_col2, submit_col3 = st.columns([1, 2, 1])
        with submit_col2:
            submit = st.form_submit_button(
                "🚀 Get Intelligent Recommendations",
                use_container_width=True,
                type="primary"
            )
    
    return {
        "submit": submit,
        "current_class": current_class,
        "stream": stream,
        "state": state,
        "board": board,
        "scholarship_needed": scholarship_needed == "Yes",
        "age": age,
        "percentage": percentage
    }

# ================= PROCESS STUDENT QUERY =================

def process_student_query_with_ml(student_data, backend, database):
    """Process student query with ML enhancements"""
    try:
        # Prepare data for backend
        backend_data = {
            "current_class": student_data["current_class"],
            "stream": student_data["stream"],
            "state": student_data["state"],
            "board": student_data["board"],
            "scholarship_needed": student_data["scholarship_needed"],
            "age": student_data["age"],
            "percentage": student_data["percentage"]
        }
        
        # Validate data
        is_valid, errors = validate_student_data(backend_data)
        if not is_valid:
            st.error(f"Validation errors: {', '.join(errors)}")
            return None, []
        
        # Get university exams matching student criteria
        uni_exams_df = database.get_exams()
        matching_uni_exams = []
        
        # Eligibility mapping
        eligibility = {
            '8': ['8', '9', '10', '11', '12', 'Diploma', 'UG', 'PG', 'PhD'],
            '9': ['9', '10', '11', '12', 'Diploma', 'UG', 'PG', 'PhD'],
            '10': ['10', '11', '12', 'Diploma', 'UG', 'PG', 'PhD'],
            '11': ['11', '12', 'Diploma', 'UG', 'PG', 'PhD'],
            '12': ['12', 'Diploma', 'UG', 'PG', 'PhD'],
            'Diploma': ['Diploma', 'UG', 'PG', 'PhD'],
            'UG': ['UG', 'PG', 'PhD'],
            'PG': ['PG', 'PhD'],
            'PhD': ['PhD']
        }
        
        student_class = student_data["current_class"]
        student_stream = student_data["stream"]
        
        if not uni_exams_df.empty:
            class_order = ['8','9','10','11','12','Diploma','UG','PG','PhD']
            student_idx = class_order.index(student_class)
            for _, row in uni_exams_df.iterrows():
                exam_class = row["eligible_from"]

                if exam_class not in class_order:
                    continue

                exam_idx = class_order.index(exam_class)

            # Stream check
                if row["stream"] != "All" and row["stream"] != student_stream:
                    continue

                exam_dict = row.to_dict()

                if student_idx >= exam_idx:
                    exam_dict["eligibility_status"] = "Currently Eligible"
                    matching_uni_exams.append(exam_dict)

                elif student_idx + 2 >= exam_idx:
                    exam_dict["eligibility_status"] = "Eligible Soon"
                    matching_uni_exams.append(exam_dict)

        
        # Get recommendations from backend
        result = backend.process_student_query(
            **backend_data,
            university_exams=matching_uni_exams
        )
        
        # Apply ML enhancements if models are trained
        if st.session_state.ml_trained and result:
            result = apply_ml_enhancements(result, backend_data)
        
        # Log the student query for analytics
        if result and 'eligible_exams_count' in result:
            database.log_student_query({
                "student_class": student_data["current_class"],
                "stream": student_data["stream"],
                "state": student_data["state"],
                "eligible_exams_count": result.get('eligible_exams_count', 0)
            })
        
        return result, matching_uni_exams
        
    except Exception as e:
        st.error(f"Error processing query: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, []

def apply_ml_enhancements(result, student_data):
    """Apply ML enhancements to recommendations"""
    if not result or 'government_exams' not in result:
        return result
    
    enhanced_exams = []
    for exam_data in result['government_exams']:
        exam_score = 0.5  # Default base score
        
        # ML prediction if model is trained
        if st.session_state.ml_trained:
            # Create exam object for ML prediction
            from dataclasses import make_dataclass
            
            # Create a simple exam-like object
            ExamSimple = make_dataclass('ExamSimple', [
                ('stream', str),
                ('level', str),
                ('eligible_from', str),
                ('scholarship_available', bool),
                ('min_percentage', float),
                ('state_specific', str)
            ])
            
            exam_obj = ExamSimple(
                stream=exam_data.get('stream', 'All'),
                level=exam_data.get('level', 'UG'),
                eligible_from=exam_data.get('eligible_from', '12'),
                scholarship_available=exam_data.get('scholarship', False),
                min_percentage=exam_data.get('min_percentage', 0.0),
                state_specific=exam_data.get('state_specific', 'All-India')
            )
            
            # Create student profile object
            StudentSimple = make_dataclass('StudentSimple', [
                ('stream', str),
                ('current_class', str),
                ('state', str),
                ('percentage', float),
                ('age', int),
                ('scholarship_needed', bool)
            ])
            
            student_profile = StudentSimple(
                stream=student_data['stream'],
                current_class=student_data['current_class'],
                state=student_data['state'],
                percentage=student_data.get('percentage', 50.0),
                age=student_data.get('age', 18),
                scholarship_needed=student_data['scholarship_needed']
            )
            
            # Get ML prediction
            try:
                ml_score = st.session_state.ml_recommender.predict_relevance(student_profile, exam_obj)
                exam_score = ml_score  # Use ML score directly
            except:
                exam_score = 0.5  # Fallback to default
        
        # Add enhanced score to exam
        exam_data['ml_enhanced_score'] = exam_score
        enhanced_exams.append(exam_data)
    
    # Sort by enhanced score
    enhanced_exams.sort(key=lambda x: x.get('ml_enhanced_score', 0.5), reverse=True)
    result['government_exams'] = enhanced_exams
    
    # Add ML metadata
    result['ml_metadata'] = {
        'ml_models_used': st.session_state.ml_trained,
        'enhancement_applied': st.session_state.ml_trained
    }

    for idx, exam in enumerate(enhanced_exams, start=1):
        exam['ml_rank'] = idx
    
    return result

# ================= DISPLAY EXAM CARD =================

def display_exam_card(exam, exam_type="Government", ml_enhanced=False):
    """Display an exam in a card format"""
    with st.container():
        # Card header with ML badge if enhanced
        ml_badge = "🤖 " if ml_enhanced and exam.get('ml_enhanced_score') else ""
        exam_name = exam.get('name') or exam.get('exam_name', 'Unknown Exam')
        
        st.markdown(f"""
        <div class="exam-card">
            <h4>{ml_badge}{exam_name}</h4>
            <p><strong>Conducted by:</strong> {exam.get('conducting_body', exam.get('university_name', 'N/A'))}</p>
        """, unsafe_allow_html=True)
        
        # Display badges
        badge_html = ""
        if exam_type == "Government":
            if exam.get('scholarship'):
                badge_html += '<span class="feature-badge">🎓 Scholarship</span>'
            if exam.get('stream'):
                badge_html += f'<span class="feature-badge">📚 {exam.get("stream")}</span>'
            
            if exam.get('level'):
                badge_html += f'<span class="feature-badge">📈 {exam.get("level")}</span>'
            if ml_enhanced and exam.get('ml_enhanced_score'):
                score = exam['ml_enhanced_score']
                color = "#4ECDC4" if score > 0.7 else "#FF6B6B" if score < 0.3 else "#FFD166"
                badge_html += f'<span class="feature-badge" style="background: {color};">🤖 ML Score: {score:.2f}</span>'
        else:
            if exam.get('scholarship') == "Yes":
                badge_html += '<span class="feature-badge">🎓 Scholarship</span>'
            if exam.get('stream'):
                badge_html += f'<span class="feature-badge">📚 {exam.get("stream")}</span>'
            if exam.get('exam_level'):
                badge_html += f'<span class="feature-badge">📈 {exam.get("exam_level")}</span>'
        if exam.get("eligibility_status"):
            badge_html += f'<span class="feature-badge">🕒 {exam["eligibility_status"]}</span>'
        st.markdown(badge_html, unsafe_allow_html=True)
        
        # Key details
        col1, col2 = st.columns(2)
        with col1:
            if exam_type == "Government":
                st.write(f"**Level:** {exam.get('level', 'N/A')}")
                st.write(f"**Eligibility:** {exam.get('eligible_from', 'N/A')}+")
                st.write(f"**Mode:** {exam.get('exam_mode', 'N/A')}")
            else:
                st.write(f"**Level:** {exam.get('exam_level', 'N/A')}")
                st.write(f"**Eligibility:** {exam.get('eligible_from', 'N/A')}+")
                st.write(f"**Mode:** {exam.get('exam_mode', 'N/A')}")
        
        with col2:
            if exam_type == "Government":
                last_date = exam.get('last_date', 'N/A')
                if last_date and last_date != 'N/A':
                    last_date = last_date[:10] if isinstance(last_date, str) and len(last_date) > 10 else last_date
                st.write(f"**Last Date:** {last_date}")
                st.write(f"**Scholarship:** {'Yes' if exam.get('scholarship') else 'No'}")
                if exam.get('min_percentage'):
                    st.write(f"**Min %:** {exam['min_percentage']}%")
            else:
                last_date = exam.get('last_date', 'N/A')
                st.write(f"**Last Date:** {last_date}")
                st.write(f"**Scholarship:** {exam.get('scholarship', 'N/A')}")
                if exam.get('min_percentage'):
                    st.write(f"**Min %:** {exam['min_percentage']}%")
        
        st.markdown("</div>", unsafe_allow_html=True)

# ================= DISPLAY RECOMMENDATIONS =================

def display_recommendations(result, university_exams):
    """Display all recommendations in an organized way"""
    
    if not result:
        st.error("No recommendations received. Please try again.")
        return
    
    # Show success message
    eligible_count = result.get('eligible_exams_count', 0)
    ml_enhanced = result.get('ml_metadata', {}).get('enhancement_applied', False)
    
    st.success(f"✅ Found {eligible_count} government exams and {len(university_exams)} university exams!")
    
    if ml_enhanced:
        st.info("🤖 Recommendations enhanced with ML models")
    
    # Tabs for different exam types
    tab1, tab2 = st.tabs(["🏛️ Government Exams", "🏫 University Exams"])
    
    with tab1:
        # Government Exams
        gov_exams = result.get("government_exams", [])
        if gov_exams:
            st.markdown(f"### Top {len(gov_exams)} Government Exam Recommendations")
            
            # Display all exams in cards
            for exam in gov_exams:
                display_exam_card(exam, "Government", ml_enhanced)
            
            # ML enhanced visualization
            if ml_enhanced and len(gov_exams) > 0:
                st.markdown("---")
                st.markdown("#### 🤖 ML Recommendation Analysis")
                
                # Create visualization of scores
                scores = [exam.get('ml_enhanced_score', 0.5) for exam in gov_exams[:10]]
                exam_names = []
                for exam in gov_exams[:10]:
                    name = exam['name']
                    exam_names.append(name[:30] + "..." if len(name) > 30 else name)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=exam_names,
                        y=scores,
                        marker_color=['#4ECDC4' if s > 0.7 else '#FFD166' if s > 0.4 else '#FF6B6B' 
                                    for s in scores],
                        text=[f"{s:.2f}" for s in scores],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title='ML Recommendation Scores',
                    xaxis_title='Exam',
                    yaxis_title='Score',
                    yaxis_range=[0, 1],
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Score interpretation
                col1, col2, col3 = st.columns(3)
                col1.metric("High Relevance", f"{(sum(1 for s in scores if s > 0.7)/len(scores)*100):.0f}%")
                col2.metric("Medium Relevance", f"{(sum(1 for s in scores if 0.4 <= s <= 0.7)/len(scores)*100):.0f}%")
                col3.metric("Low Relevance", f"{(sum(1 for s in scores if s < 0.4)/len(scores)*100):.0f}%")
            
            # Show all details in expander
            with st.expander("📋 View All Details in Table Format"):
                gov_df = pd.DataFrame(gov_exams)
                st.dataframe(gov_df, use_container_width=True)
        else:
            st.info("No government exams found matching your criteria.")
    
    with tab2:
        # University Exams
        if university_exams:
            st.markdown(f"### Top {len(university_exams)} University Exam Matches")
            
            for exam in university_exams:
                display_exam_card(exam, "University")
            
            with st.expander("📋 View All University Exam Details"):
                uni_df = pd.DataFrame(university_exams)
                st.dataframe(uni_df, use_container_width=True)
        else:
            st.info("No university exams found matching your criteria.")
    
# ================= UNIVERSITY PORTAL =================

def render_university_portal():
    """Render the university exam registration portal"""
    st.markdown("## 🏫 University Exam Registration Portal")
    st.markdown("Universities can register their entrance exams here to reach eligible students.")
    
    with st.form("university_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            university_name = st.text_input("University/Institution Name *")
            exam_name = st.text_input("Exam Name *")
            exam_level = st.selectbox("Exam Level *", ["School", "UG", "PG", "PhD", "Diploma"])
            stream = st.selectbox("Eligible Stream *", 
                                ["All", "Science (PCM)", "Science (PCB)", "Science (PCMB)", 
                                 "Commerce", "Arts/Humanities", "Law", "Design", 
                                 "Architecture", "Management", "IT / Computer Science"])
        
        with col2:
            eligible_from = st.selectbox("Eligible From Class/Level *", 
                                       ["8", "9", "10", "11", "12", "Diploma", "UG", "PG"])
            last_date = st.date_input("Last Date to Apply *", min_value=date.today())
            contact_email = st.text_input("Contact Email *")
            website = st.text_input("Official Website")
            scholarship = st.selectbox("Scholarship Available *", ["Yes", "No", "Limited"])
            exam_mode = st.selectbox("Exam Mode *", ["Online", "Offline", "Hybrid"])
        
        # Additional details
        min_percentage = st.slider("Minimum Percentage Required", 0, 100, 50, 1)
        application_fee = st.number_input("Application Fee (₹)", min_value=0, value=0, step=100)
        
        col3, col4 = st.columns(2)
        with col3:
            state = st.text_input("State (optional)")
        with col4:
            city = st.text_input("City (optional)")
        
        syllabus_link = st.text_input("Syllabus/Pattern Link (optional)")
        
        # Submit button
        submit = st.form_submit_button("📝 Register Exam", type="primary")
    
    if submit:
        if not all([university_name, exam_name, contact_email]):
            st.error("Please fill all required fields (*)")
        else:
            database = get_database_manager(use_sqlite=False)
            success, message, exam_id = database.add_exam({
                "university_name": university_name,
                "exam_name": exam_name,
                "exam_level": exam_level,
                "eligible_from": eligible_from,
                "stream": stream,
                "age_limit": "None",
                "exam_mode": exam_mode,
                "scholarship": scholarship,
                "last_date": last_date.isoformat(),
                "contact_email": contact_email,
                "website": website if website else None,
                "min_percentage": float(min_percentage),
                "application_fee": float(application_fee) if application_fee > 0 else None,
                "state": state if state else None,
                "city": city if city else None,
                "syllabus_link": syllabus_link if syllabus_link else None
            })
            
            if success:
                st.success(f"✅ {message}")
                st.info(f"**Exam ID:** {exam_id}")
                st.balloons()
            else:
                st.error(f"❌ {message}")
    
    # View registered exams
    st.markdown("---")
    st.markdown("### 📋 Currently Registered Exams")
    
    database = get_database_manager(use_sqlite=False)
    exams_df = database.get_exams()
    
    if not exams_df.empty:
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            filter_level = st.selectbox("Filter by Level", 
                                      ["All"] + ["School", "UG", "PG", "PhD", "Diploma"])
        
        # Apply filters
        filtered_df = exams_df.copy()
        if filter_level != "All":
            filtered_df = filtered_df[filtered_df['exam_level'] == filter_level]
        
        st.dataframe(filtered_df, use_container_width=True)
        
        # Export option
        if st.button("📥 Export to CSV"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV File",
                data=csv,
                file_name="registered_exams.csv",
                mime="text/csv"
            )
    else:
        st.info("No exams registered yet. Be the first to add an exam!")

# ================= MAIN APP =================

def main():
    """Main application function"""
    # Initialize
    load_custom_css()
    initialize_session_state()
    
    # Render header
    render_header()
    
    # ML Training Section
    render_ml_training_section()
    
    # Create tabs for main sections
    tab1, tab2 = st.tabs(["🎓 Student Portal", "🏫 University Portal"])
    
    with tab1:
        # Get backend and database instances
        backend, database = initialize_systems()
        
        # Render student form
        form_data = render_student_form()
        
        # Process form submission
        if form_data["submit"]:
            with st.spinner("🔍 Analyzing your profile with ML algorithms..."):
                result, university_exams = process_student_query_with_ml(
                    form_data, 
                    backend, 
                    database
                )
                
                if result:
                    st.session_state.recommendations = result
                    st.session_state.university_exams = university_exams
                    st.session_state.student_profile = form_data
                else:
                    st.error("Failed to get recommendations. Please try again.")
        
        # Display recommendations if available
        if st.session_state.recommendations:
            st.markdown("---")
            display_recommendations(
                st.session_state.recommendations, 
                st.session_state.university_exams
            )

            st.markdown("## 🧠 Personalized AI Guidance")

            with st.spinner("🤖 Generating expert guidance using AI..."):
                ai_guidance = generate_personalized_guidance(
                    st.session_state.student_profile,
                    st.session_state.recommendations
                )

            st.markdown(ai_guidance)

    
    with tab2:
        render_university_portal()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 2rem 0;'>
            <p>© 2026 ExamGenius </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ================= RUN APP =================
if __name__ == "__main__":
    main()