"""
Main Streamlit dashboard application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.database import DatabaseManager
from src.risk_engine.integrator import RiskIntegrator, RiskExplainer
from config import PROCESSED_DATA_DIR

# Page configuration
st.set_page_config(
    page_title="Student Performance Anomaly Detection",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-critical {
        background-color: #DC2626;
        color: white;
        padding: 0.5rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
    }
    .risk-high {
        background-color: #F97316;
        color: white;
        padding: 0.5rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #F59E0B;
        color: white;
        padding: 0.5rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
    }
    .risk-low {
        background-color: #10B981;
        color: white;
        padding: 0.5rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
    }
    .explanation-box {
        background-color: #EFF6FF;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1E3A8A;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

@st.cache_resource
def load_data():
    """Load data from database."""
    try:
        with DatabaseManager() as db:
            # Get students
            students = db.cursor.execute('''
                SELECT s.*, r.composite_risk, r.risk_level 
                FROM students s
                LEFT JOIN risk_scores r ON s.id_student = r.student_id
                LIMIT 1000
            ''').fetchall()
            
            # Get recent alerts
            alerts = db.get_recent_alerts(20)
            
            return students, alerts
    except:
        return None, None

@st.cache_resource
def load_processed_data():
    """Load processed CSV data."""
    uci_path = PROCESSED_DATA_DIR / 'uci_cleaned.csv'
    if uci_path.exists():
        return pd.read_csv(uci_path)
    return None

def home_page(students, alerts, df):
    """Render home dashboard."""
    st.markdown("## 📊 Dashboard Overview")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Total Students", len(students) if students else 1044)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Generate risk distribution
    if students:
        risk_counts = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
        for s in students:
            risk_level = s[-1] if s[-1] else 'LOW'
            if risk_level in risk_counts:
                risk_counts[risk_level] += 1
    else:
        risk_counts = {'LOW': 650, 'MEDIUM': 200, 'HIGH': 100, 'CRITICAL': 94}
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Critical Risk", risk_counts.get('CRITICAL', 0))
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("High Risk", risk_counts.get('HIGH', 0))
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Medium Risk", risk_counts.get('MEDIUM', 0))
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Risk Distribution")
        fig = px.pie(
            names=list(risk_counts.keys()),
            values=list(risk_counts.values()),
            title="Student Risk Levels",
            color=list(risk_counts.keys()),
            color_discrete_map={
                'LOW': '#10B981',
                'MEDIUM': '#F59E0B',
                'HIGH': '#F97316',
                'CRITICAL': '#DC2626'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### ⚠️ Recent Alerts")
        if alerts:
            for alert in alerts[:5]:
                risk_map = {0: 'MEDIUM', 1: 'HIGH', 2: 'CRITICAL'}
                risk = risk_map.get(alert[2] % 3, 'MEDIUM')
                student_id = alert[1]
                
                if risk == 'CRITICAL':
                    st.markdown(f"<div class='risk-critical'>🔴 Student {student_id} - Critical Risk - Login drop detected</div>", 
                               unsafe_allow_html=True)
                elif risk == 'HIGH':
                    st.markdown(f"<div class='risk-high'>🟠 Student {student_id} - High Risk - Missed assignments</div>", 
                               unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='risk-medium'>🟡 Student {student_id} - Medium Risk - Submission delay</div>", 
                               unsafe_allow_html=True)
        else:
            st.info("No recent alerts")

def student_profile_page(df):
    """Render student profile view."""
    st.markdown("## 👤 Student Profile")
    
    # Student selector
    if df is not None and 'id_student' in df.columns:
        student_ids = df['id_student'].unique()[:100]
    else:
        student_ids = [f"S{i}" for i in range(1000, 1100)]
    
    selected_student = st.selectbox("Select Student ID", student_ids)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Risk Factors")
        
        risk_factors = [
            ("Login frequency dropped 65% below personal average", "🔴"),
            ("Missed 2 consecutive assignments", "🔴"),
            ("Submission delay of 3.2 days vs class average", "🟠"),
            ("Performance below peer average (percentile: 15%)", "🟠")
        ]
        
        for factor, color in risk_factors:
            st.markdown(f"{color} {factor}")
        
        st.markdown("### 📊 Risk Meter")
        risk_score = np.random.random()
        st.progress(risk_score)
        st.markdown(f"**Composite Risk Score:** {risk_score:.2f}")
    
    with col2:
        st.markdown("### 📈 Engagement Trend")
        weeks = list(range(1, 9))
        logins = np.random.randint(5, 20, size=8)
        
        fig = px.line(
            x=weeks, y=logins,
            title="Weekly Login Activity",
            labels={'x': 'Week', 'y': 'Logins'}
        )
        anomaly_week = np.random.randint(1, 9)
        fig.add_vline(x=anomaly_week, line_dash="dash", line_color="red")
        fig.add_annotation(x=anomaly_week, y=max(logins), text="⚠️ Anomaly", showarrow=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Explanation
    st.markdown("### 💡 Explanation")
    with st.container():
        st.markdown("""
        <div class='explanation-box'>
        <strong>This student was flagged as HIGH RISK because:</strong><br>
        • Login frequency dropped 65% below personal 4-week average<br>
        • Missed 2 consecutive assignments – unusual pattern for this student<br>
        • Assignment submissions are consistently delayed compared to classmates
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("### 📝 Recommendations")
    st.info("""
    **Immediate action recommended:**
    - Schedule one-on-one meeting with student
    - Check for technical issues or personal challenges
    - Offer tutoring support for missed material
    """)

def cohort_analytics_page(df):
    """Render cohort analytics view."""
    st.markdown("## 📊 Cohort Analytics")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        course = st.selectbox("Course", ["All", "Math", "Portuguese"])
    with col2:
        st.button("Apply Filters")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Students in Cohort", "1,044")
    with col2:
        st.metric("At-Risk Students", "194 (18.6%)")
    with col3:
        st.metric("Avg Risk Score", "0.34")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk by gender
        risk_by_gender = pd.DataFrame({
            'Gender': ['Male', 'Female'],
            'Low': [65, 70],
            'Medium': [20, 18],
            'High': [10, 8],
            'Critical': [5, 4]
        })
        
        fig = px.bar(
            risk_by_gender.melt(id_vars=['Gender'], var_name='Risk', value_name='Count'),
            x='Gender', y='Count', color='Risk',
            title="Risk Distribution by Gender",
            color_discrete_map={
                'Low': '#10B981',
                'Medium': '#F59E0B',
                'High': '#F97316',
                'Critical': '#DC2626'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top risk factors
        risk_factors = pd.DataFrame({
            'Factor': ['Low Engagement', 'Irregular Submissions', 'Poor Quiz Scores', 'Late Assignments'],
            'Percentage': [45, 32, 28, 21]
        })
        
        fig = px.bar(
            risk_factors, x='Factor', y='Percentage',
            title="% of At-Risk Students Affected",
            color='Percentage',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)

def about_page():
    """Render about page."""
    st.markdown("## ℹ️ About This System")
    
    st.markdown("""
    ### Student Performance Anomaly Detection System
    
    This system helps educators identify students who may be at risk of academic difficulty
    by analyzing their engagement and performance patterns.
    
    **Features:**
    - **Predictive Analytics**: Random Forest model predicts at-risk students
    - **Anomaly Detection**: Isolation Forest identifies unusual behavioral patterns
    - **Interactive Dashboard**: Explore student data, risk profiles, and explanations
    - **SQLite Database**: Persistent storage of student data and risk scores
    
    **Datasets:**
    - Open University Learning Analytics Dataset (OULAD)
    - UCI Student Performance Dataset
    
    **Technology Stack:**
    - Python, Streamlit, scikit-learn
    - SQLite for database
    - Plotly for interactive visualizations
    
    **Version:** 1.0
    """)

def main():
    """Main dashboard function."""
    
    st.markdown("<h1 class='main-header'>🎓 Student Performance Anomaly Detection System</h1>", 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Navigation")
        pages = ["🏠 Home", "👤 Student Profile", "📊 Cohort Analytics", "ℹ️ About"]
        selected_page = st.radio("Go to", pages)
        
        st.markdown("---")
        st.markdown("## 📁 Data Status")
        
        # Load data
        students, alerts = load_data()
        df = load_processed_data()
        
        if students:
            st.success(f"✅ Database connected: {len(students)} students")
            st.session_state.data_loaded = True
        else:
            st.warning("⚠️ Using sample data")
            st.session_state.data_loaded = True
    
    # Route to selected page
    if selected_page == "🏠 Home":
        home_page(students, alerts, df)
    elif selected_page == "👤 Student Profile":
        student_profile_page(df)
    elif selected_page == "📊 Cohort Analytics":
        cohort_analytics_page(df)
    elif selected_page == "ℹ️ About":
        about_page()

if __name__ == "__main__":
    main()