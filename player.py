import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Player Engagement Predictor",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state for theme
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

def get_theme_css():
    if st.session_state.dark_mode:
        return """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Roboto:wght@300;400;500;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #ffffff;
    }
    
    .main {
        background: rgba(26, 26, 46, 0.95);
        background-attachment: fixed;
        min-height: 100vh;
        padding: 0;
        color: #ffffff;
    }
    
    .main > div {
        background: rgba(22, 33, 62, 0.9);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    h1, h2, h3 {
        font-family: 'Orbitron', monospace;
        color: #ffffff;
        text-shadow: 0 0 10px rgba(79, 172, 254, 0.5);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: none;
        filter: drop-shadow(0 0 20px rgba(79, 172, 254, 0.5));
    }
    
    .main-title .emoji {
        font-size: 3.5rem;
        margin-right: 1rem;
        filter: none;
        color: initial;
    }
    
    .main-title .text {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        display: inline-block;
    }
    
    .subtitle {
        text-align: center;
        color: #b0b0b0;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }
    
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        text-align: center;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .theme-toggle-container {
        display: flex;
        justify-content: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .prediction-high {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.4);
        margin: 2rem 0;
        border: 3px solid rgba(255, 255, 255, 0.2);
    }
    
    .prediction-medium {
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(255, 152, 0, 0.4);
        margin: 2rem 0;
        border: 3px solid rgba(255, 255, 255, 0.2);
    }
    
    .prediction-low {
        background: linear-gradient(135deg, #f44336 0%, #da190b 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(244, 67, 54, 0.4);
        margin: 2rem 0;
        border: 3px solid rgba(255, 255, 255, 0.2);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 2rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .stSelectbox > div > div {
        background: rgba(22, 33, 62, 0.9);
        border-radius: 10px;
        border: 2px solid rgba(79, 172, 254, 0.3);
        color: #ffffff;
    }
    
    .stNumberInput > div > div {
        background: rgba(22, 33, 62, 0.9);
        border-radius: 10px;
        border: 2px solid rgba(79, 172, 254, 0.3);
        color: #ffffff;
    }
    
    .stSlider > div > div {
        background: rgba(22, 33, 62, 0.9);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(79, 172, 254, 0.3);
    }
    
    .insight-item {
        background: rgba(79, 172, 254, 0.2);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4facfe;
        color: #ffffff;
    }
    
    .probability-card {
        background: rgba(22, 33, 62, 0.9);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border: 2px solid rgba(79, 172, 254, 0.3);
        color: #ffffff;
    }
    
    .gaming-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    
    .divider {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        height: 3px;
        border-radius: 2px;
        margin: 2rem 0;
    }
    
    /* Dark mode specific overrides */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #ffffff !important;
    }
    
    .stMarkdown {
        color: #ffffff;
    }
    
    div[data-testid="stExpander"] {
        background: rgba(22, 33, 62, 0.9);
        border: 1px solid rgba(79, 172, 254, 0.3);
        border-radius: 10px;
    }
    
    div[data-testid="stExpander"] summary {
        color: #ffffff;
    }
</style>
"""
    else:
        return """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Roboto:wght@300;400;500;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #333333;
    }
    
    .main {
        background: rgba(255, 255, 255, 0.95);
        background-attachment: fixed;
        min-height: 100vh;
        padding: 0;
        color: #333333;
    }
    
    .main > div {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    h1, h2, h3 {
        font-family: 'Orbitron', monospace;
        color: #2c3e50;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: none;
    }
    
    .main-title .emoji {
        font-size: 3.5rem;
        margin-right: 1rem;
        filter: none;
        color: initial;
    }
    
    .main-title .text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        display: inline-block;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }
    
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        text-align: center;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .theme-toggle-container {
        display: flex;
        justify-content: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .prediction-high {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.3);
        margin: 2rem 0;
        border: 3px solid rgba(255, 255, 255, 0.2);
    }
    
    .prediction-medium {
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(255, 152, 0, 0.3);
        margin: 2rem 0;
        border: 3px solid rgba(255, 255, 255, 0.2);
    }
    
    .prediction-low {
        background: linear-gradient(135deg, #f44336 0%, #da190b 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(244, 67, 54, 0.3);
        margin: 2rem 0;
        border: 3px solid rgba(255, 255, 255, 0.2);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 2rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    .stNumberInput > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    .stSlider > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 1rem;
    }
    
    .insight-item {
        background: rgba(102, 126, 234, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        color: #333333;
    }
    
    .probability-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: 2px solid rgba(102, 126, 234, 0.2);
        color: #333333;
    }
    
    .gaming-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    
    .divider {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        height: 3px;
        border-radius: 2px;
        margin: 2rem 0;
    }
</style>
"""

# Load models and scaler
@st.cache_resource
def load_models():
    try:
        model = joblib.load('OP_model.pkl')
        scaler = joblib.load('OP_scaler.pkl')
        numeric_features = joblib.load('numeric_features.pkl')
        return model, scaler, numeric_features
    except FileNotFoundError:
        st.error("🚨 Model files not found. Please make sure 'OP_model.pkl', 'OP_scaler.pkl', and 'numeric_features.pkl' are in the same directory.")
        return None, None, None

def create_feature_vector(user_input, numeric_features):
    """Create feature vector for prediction"""
    # Create base dataframe with all possible features
    feature_dict = {}
    
    # Add numeric features (excluding PlayerID as it's just an identifier)
    for feature in numeric_features:
        if feature in user_input and feature != 'PlayerID':
            feature_dict[feature] = user_input[feature]
    
    # Add location features (one-hot encoded) - Only Europe and Other (USA is reference)
    feature_dict['Location_Europe'] = 1 if user_input['Location'] == 'Europe' else 0
    feature_dict['Location_Other'] = 1 if user_input['Location'] == 'Other' else 0
    
    # Add game genre features (one-hot encoded) - Only RPG, Simulation, Sports, Strategy (Action is reference)
    feature_dict['GameGenre_RPG'] = 1 if user_input['GameGenre'] == 'RPG' else 0
    feature_dict['GameGenre_Simulation'] = 1 if user_input['GameGenre'] == 'Simulation' else 0
    feature_dict['GameGenre_Sports'] = 1 if user_input['GameGenre'] == 'Sports' else 0
    feature_dict['GameGenre_Strategy'] = 1 if user_input['GameGenre'] == 'Strategy' else 0
    
    # Add game difficulty features (one-hot encoded) - Only Hard and Medium (Easy is reference)
    feature_dict['GameDifficulty_Hard'] = 1 if user_input['GameDifficulty'] == 'Hard' else 0
    feature_dict['GameDifficulty_Medium'] = 1 if user_input['GameDifficulty'] == 'Medium' else 0
    
    # Add derived features (these should match your model's expected features)
    feature_dict['EngagementPerSession'] = user_input['PlayTimeHours'] * 60 / (user_input['SessionsPerWeek'] * 4) if user_input['SessionsPerWeek'] > 0 else 0
    feature_dict['WeeklyPlayTime'] = user_input['PlayTimeHours'] / 4  # Assuming monthly data
    
    # Add intensity score (you may need to adjust this calculation based on your model)
    feature_dict['IntensityScore'] = (user_input['PlayTimeHours'] * user_input['SessionsPerWeek'] * user_input['AvgSessionDurationMinutes']) / 1000
    
    # Add InGamePurchases if it was part of the original features
    if 'InGamePurchases' in user_input:
        feature_dict['InGamePurchases'] = user_input['InGamePurchases']
    
    return pd.DataFrame([feature_dict])

def get_model_expected_features(model):
    """Get the expected feature order from the model"""
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    elif hasattr(model, 'n_features_in_'):
        return [f'feature_{i}' for i in range(model.n_features_in_)]
    else:
        return None

def align_features_with_model(feature_df, model, expected_features=None):
    """Align feature dataframe with model's expected features"""
    if expected_features is None:
        expected_features = get_model_expected_features(model)
    
    if expected_features is None:
        return feature_df
    
    # Create a new dataframe with the expected features
    aligned_df = pd.DataFrame(columns=expected_features)
    
    # Fill in the features that exist
    for col in expected_features:
        if col in feature_df.columns:
            aligned_df[col] = feature_df[col]
        else:
            aligned_df[col] = 0  # Default value for missing features
    
    return aligned_df

def main():
    # Apply theme CSS
    st.markdown(get_theme_css(), unsafe_allow_html=True)
    
    # Main title with enhanced styling
    st.markdown('<h1 class="main-title"><span class="emoji">🎮</span><span class="text">Player Engagement Predictor</span></h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced AI-powered player engagement analysis and prediction system</p>', unsafe_allow_html=True)
    
    # Theme toggle button in the main area
    st.markdown('<div class="theme-toggle-container">', unsafe_allow_html=True)
    theme_icon = "🌙" if not st.session_state.dark_mode else "☀️"
    theme_text = "Switch to Dark Mode" if not st.session_state.dark_mode else "Switch to Light Mode"
    
    if st.button(f"{theme_icon} {theme_text}", key="theme_toggle"):
        toggle_theme()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Load models
    model, scaler, numeric_features = load_models()
    
    if model is None:
        st.stop()
    
    # Create two columns for input
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="section-header"><div class="gaming-icon">🎯</div>Basic Player Information</div>', unsafe_allow_html=True)
        
        # Player ID
        player_id = st.number_input("🆔 Player ID", min_value=1, value=90001, step=1)
        
        # Age
        age = st.slider("🎂 Age", min_value=18, max_value=65, value=30)
        
        # Gender
        gender = st.selectbox("👤 Gender", ["Male", "Female"])
        
        # Location
        location = st.selectbox("🌍 Location", ["USA", "Europe", "Other"])
        
        # Game Genre
        game_genre = st.selectbox("🎮 Game Genre", 
                                 ["Action", "Strategy", "Sports", "RPG", "Simulation"])
        
        # Game Difficulty
        game_difficulty = st.selectbox("⚡ Game Difficulty", 
                                     ["Easy", "Medium", "Hard"])
    
    with col2:
        st.markdown('<div class="section-header"><div class="gaming-icon">📊</div>Gaming Statistics</div>', unsafe_allow_html=True)
        
        # Play Time Hours
        play_time_hours = st.number_input("⏱️ Play Time Hours", 
                                        min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        
        # In Game Purchases - Now as binary choice
        in_game_purchases = st.selectbox("💰 In Game Purchases", 
                                       options=[0, 1],
                                       format_func=lambda x: "No Purchases" if x == 0 else "Has Purchases")
        
        # Sessions Per Week
        sessions_per_week = st.slider("📅 Sessions Per Week", 
                                    min_value=1, max_value=20, value=5)
        
        # Average Session Duration
        avg_session_duration = st.slider("⏰ Average Session Duration (Minutes)", 
                                       min_value=10, max_value=300, value=60)
        
        # Player Level
        player_level = st.number_input("🏆 Player Level", 
                                     min_value=1, max_value=100, value=25)
        
        # Achievements Unlocked
        achievements_unlocked = st.number_input("🎖️ Achievements Unlocked", 
                                              min_value=0, max_value=100, value=10)
    
    # Enhanced prediction button
    if st.button("🔮 Predict Engagement Level"):
        # Prepare input data (excluding PlayerID as it's just an identifier)
        user_input = {
            'Age': age,
            'Gender': gender,
            'Location': location,
            'GameGenre': game_genre,
            'PlayTimeHours': play_time_hours,
            'InGamePurchases': in_game_purchases,
            'GameDifficulty': game_difficulty,
            'SessionsPerWeek': sessions_per_week,
            'AvgSessionDurationMinutes': avg_session_duration,
            'PlayerLevel': player_level,
            'AchievementsUnlocked': achievements_unlocked
        }
        
        try:
            # Create feature vector
            feature_df = create_feature_vector(user_input, numeric_features)
            
            # Align features with model expectations
            aligned_df = align_features_with_model(feature_df, model)
            
            # Debug information
            with st.expander("🔍 Debug Information"):
                st.write(f"**Original features count:** {len(feature_df.columns)}")
                st.write(f"**Aligned features count:** {len(aligned_df.columns)}")
                st.write(f"**Model expected features:** {model.n_features_in_ if hasattr(model, 'n_features_in_') else 'Unknown'}")
                st.write("**Feature columns:**", list(aligned_df.columns))
            
            # Scale only the numeric features that exist in the aligned dataframe
            aligned_df_scaled = aligned_df.copy()
            # Filter out PlayerID from numeric features if it exists
            numeric_cols_to_scale = [col for col in numeric_features if col in aligned_df.columns and col != 'PlayerID']
            
            if numeric_cols_to_scale:
                aligned_df_scaled[numeric_cols_to_scale] = scaler.transform(aligned_df[numeric_cols_to_scale])
            
            # Make prediction
            prediction = model.predict(aligned_df_scaled)[0]
            prediction_proba = model.predict_proba(aligned_df_scaled)[0] if hasattr(model, 'predict_proba') else None
            
            # Display divider
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            # Display results
            st.markdown("## 📈 Prediction Results")
            
            # Map prediction to engagement level
            engagement_levels = {1: "Low", 2: "Medium", 0: "High"}
            predicted_level = engagement_levels.get(prediction, "Unknown")
            
            # Display prediction with enhanced styling
            if predicted_level == "High":
                st.markdown(f"""
                <div class="prediction-high">
                    <div class="gaming-icon">🔥</div>
                    <h2>High Engagement Level</h2>
                    <p style="font-size: 1.2rem;">This player is highly engaged and likely to continue playing actively!</p>
                </div>
                """, unsafe_allow_html=True)
            elif predicted_level == "Medium":
                st.markdown(f"""
                <div class="prediction-medium">
                    <div class="gaming-icon">⚡</div>
                    <h2>Medium Engagement Level</h2>
                    <p style="font-size: 1.2rem;">This player shows moderate engagement with room for improvement.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-low">
                    <div class="gaming-icon">📉</div>
                    <h2>Low Engagement Level</h2>
                    <p style="font-size: 1.2rem;">This player may need incentives to increase engagement.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display probabilities if available
            if prediction_proba is not None:
                st.markdown("## 🎯 Prediction Probabilities")
                prob_col1, prob_col2, prob_col3 = st.columns(3)
                
                with prob_col1:
                    st.markdown(f"""
                    <div class="probability-card">
                        <h3>🔥 High Engagement</h3>
                        <h2 style="color: #4CAF50;">{prediction_proba[0]:.2%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with prob_col2:
                    st.markdown(f"""
                    <div class="probability-card">
                        <h3>⚡ Medium Engagement</h3>
                        <h2 style="color: #FF9800;">{prediction_proba[2]:.2%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with prob_col3:
                    st.markdown(f"""
                    <div class="probability-card">
                        <h3>📉 Low Engagement</h3>
                        <h2 style="color: #f44336;">{prediction_proba[1]:.2%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Additional insights
            st.markdown("## 💡 Player Insights")
            
            insights = []
            if play_time_hours > 20:
                insights.append("🎮 High play time indicates strong game interest")
            if sessions_per_week > 10:
                insights.append("📅 Frequent sessions show consistent engagement")
            if avg_session_duration > 120:
                insights.append("⏱️ Long sessions indicate deep engagement")
            if achievements_unlocked > 50:
                insights.append("🏆 High achievement count shows goal-oriented play")
            if in_game_purchases == 1:
                insights.append("💰 In-game purchases indicate investment in the game")
            
            if insights:
                for insight in insights:
                    st.markdown(f'<div class="insight-item">{insight}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="insight-item">📊 Player shows typical engagement patterns</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.error("Please check that all model files are properly saved and compatible.")

if __name__ == "__main__":
    main()
