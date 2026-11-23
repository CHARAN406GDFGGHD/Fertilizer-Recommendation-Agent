"""
UI Helpers Module
Provides shared styling, header rendering, metric cards, and sidebar components
for the Streamlit application.
"""

import os
import streamlit as st
import json
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def apply_custom_css():
    """Injects high-quality modern CSS for a clean, agricultural light theme."""
    st.markdown("""
    <style>
        /* Base typography and background */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Main container padding */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }
        
        /* Modern Card Styling */
        .agri-card {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 24px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
            margin-bottom: 20px;
            transition: all 0.2s ease-in-out;
        }
        .agri-card:hover {
            box-shadow: 0 6px 18px rgba(46, 125, 50, 0.08);
            border-color: #C8E6C9;
        }
        
        /* Metric Badge Card */
        .metric-box {
            background: linear-gradient(135deg, #F1F8E9 0%, #FFFFFF 100%);
            border: 1px solid #C8E6C9;
            border-radius: 12px;
            padding: 18px 20px;
            text-align: left;
            margin-bottom: 15px;
        }
        .metric-title {
            color: #4A5568;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }
        .metric-value {
            color: #1B5E20;
            font-size: 1.85rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .metric-subtitle {
            color: #68D391;
            font-size: 0.8rem;
            margin-top: 4px;
        }
        
        /* Recommendation Highlight Banner */
        .rec-banner {
            background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%);
            color: #FFFFFF !important;
            padding: 24px 28px;
            border-radius: 14px;
            box-shadow: 0 8px 20px rgba(46, 125, 50, 0.2);
            margin: 20px 0;
        }
        .rec-banner h2, .rec-banner h3, .rec-banner p {
            color: #FFFFFF !important;
            margin: 0;
        }
        
        /* Status Badges */
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .status-completed {
            background-color: #E8F5E9;
            color: #2E7D32;
            border: 1px solid #A5D6A7;
        }
        .status-pending {
            background-color: #FFF9C4;
            color: #F57F17;
            border: 1px solid #FFF59D;
        }
        
        /* Buttons */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            border: none;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
        }
        
        /* Table styling */
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
        }
    </style>
    """, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, icon: str = "🌱"):
    """Renders a styled page title and subtitle."""
    apply_custom_css()
    st.markdown(f"""
    <div style="margin-bottom: 24px;">
        <h1 style="color: #1B5E20; font-weight: 800; font-size: 2.1rem; margin-bottom: 6px; display: flex; align-items: center; gap: 10px;">
            <span>{icon}</span> {title}
        </h1>
        <p style="color: #556B2F; font-size: 1.05rem; margin: 0;">
            {subtitle}
        </p>
    </div>
    <hr style="border: 0; height: 1px; background-color: #E2E8F0; margin-bottom: 24px;">
    """, unsafe_allow_html=True)


def render_sidebar():
    """Renders the common project sidebar with status metrics."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 10px 0 16px 0;">
            <div style="font-size: 2.5rem; line-height: 1;">🌾</div>
            <h3 style="color: #1B5E20; margin: 6px 0 0 0; font-weight: 700; font-size: 1.15rem;">AgriFertilizer Agent</h3>
            <p style="color: #66BB6A; font-size: 0.8rem; margin: 0; font-weight: 500;">B.Tech Data Science Project</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Load metadata if present
        meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            st.markdown(f"""
            <div style="background-color: #F1F8E9; border-radius: 8px; padding: 12px; border: 1px solid #C8E6C9; margin-bottom: 14px;">
                <div style="font-size: 0.75rem; color: #4A5568; font-weight: 600;">ACTIVE ML MODEL</div>
                <div style="font-size: 0.95rem; color: #1B5E20; font-weight: 700;">{meta.get('best_model_name', 'XGBoost')}</div>
                <div style="font-size: 0.75rem; color: #2E7D32;">Test Accuracy: <b>{meta.get('best_accuracy', 1.0)*100:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("⚠️ Models not trained yet.")

        st.markdown("#### 🧭 Quick Guide")
        st.caption("1. **Recommend**: Input N-P-K & soil to get optimal fertilizer.")
        st.caption("2. **Schedule**: Build split fertilization timeline.")
        st.caption("3. **Farm DB**: Log & track farm parcels and treatments.")
        st.caption("4. **AI Assistant**: Offline agricultural Q&A agent.")
        st.caption("5. **ML Performance**: Compare 5 models & confusion matrices.")

        st.markdown("---")
        st.caption("🔒 100% Offline | Verified on Kaggle Dataset")
