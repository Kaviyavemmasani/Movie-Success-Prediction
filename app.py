import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

st.set_page_config(page_title="Movie Success Predictor", layout="wide")
@st.cache_resource
def load_model_assets():
    try:
        model = load('movie_success.joblib')
        scaler = load('scaler.joblib')
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_model_assets()

st.title("Movie Success Predictor")
st.markdown("Enter movie details..")

if model is None or scaler is None:
    st.error("**Error:** Could not find `movie_success_model.joblib` or `scaler.joblib`.")
    st.info("Please run the export cells in your Jupyter Notebook.")
else:
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Financials")
        budget = st.number_input("Budget", min_value=0, value=10000000, help="Total production budget")
        gross = st.number_input("Gross Revenue", min_value=0, value=50000000, help="Total box office earnings")
        
    with col2:
        st.subheader("Production")
        duration = st.slider("Duration (mins)", 45, 300, 120)
        title_year = st.number_input("Release Year", 1900, 2026, 2016)
        aspect_ratio = st.selectbox("Aspect Ratio", [1.85, 2.35, 1.33, 1.78])

    with col3:
        st.subheader("Social & Engagement")
        voted_users = st.number_input("Voted Users", min_value=0, value=10000)
        critic_reviews = st.number_input("Critic Reviews", min_value=0, value=100)
        user_reviews = st.number_input("User Reviews", min_value=0, value=200)
        facenumber = st.number_input("Faces on Poster", min_value=0, value=1)

    st.expander("Advanced Social Media Stats").write("Adjust FB Likes if known:")
    c1, c2, c3 = st.columns(3)
    with c1:
        director_likes = st.number_input("Director FB Likes", 0, 50000, 500)
    with c2:
        movie_likes = st.number_input("Movie FB Likes", 0, 500000, 1000)
    with c3:
        cast_total_likes = st.number_input("Total Cast FB Likes", 0, 1000000, 5000)

    st.divider()

    if st.button("Predict Success", use_container_width=True):
        input_dict = {
            'num_critic_for_reviews': critic_reviews,
            'duration': duration,
            'director_facebook_likes': director_likes,
            'actor_3_facebook_likes': 0, 
            'actor_1_facebook_likes': 0,
            'facenumber_in_poster': facenumber,
            'num_user_for_reviews': user_reviews,
            'title_year': title_year,
            'actor_2_facebook_likes': 0,
            'aspect_ratio': aspect_ratio,
            'movie_facebook_likes': movie_likes,
            'cast_total_facebook_likes': cast_total_likes, # Fixed name
            'gross_log': np.log1p(gross),
            'budget_log': np.log1p(budget),
            'num_voted_users_log': np.log1p(voted_users),
            'profit': (gross - budget) / (budget + 1e-6),
            'success_ratio': gross / (budget + 1e-6),
            'review_ratio': user_reviews / (critic_reviews + 1e-6)
        }

        df_input = pd.DataFrame([input_dict])
        try:
            trained_features = scaler.feature_names_in_
            df_final = df_input.reindex(columns=trained_features, fill_value=0)
            scaled_data = scaler.transform(df_final)
            prediction = model.predict(scaled_data)[0]
            probs = model.predict_proba(scaled_data)[0] 

            if gross < (budget * 0.5) and prediction == 2:
                result = "Flop"
            else:
                label_map = {0: "Flop", 1: "Average", 2: "Hit"}
                result = label_map[prediction]
            st.write(f"**Prediction:** {result}")
        except Exception as e:
            st.error(f"Logic Error: {e}")
            st.info("Ensure the scaler was saved.")