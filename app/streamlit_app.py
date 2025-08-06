import streamlit as st
import sklearn
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define a single, persistent location for NLTK data
NLTK_PATH = os.path.join("app", "nltk_data")
os.makedirs(NLTK_PATH, exist_ok=True)

# Set both runtime path and environment variable
nltk.data.path.append(NLTK_PATH)
os.environ["NLTK_DATA"] = NLTK_PATH

# List of required resources
REQUIRED_NLTK_RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("tokenizers/punkt_tab", "punkt_tab"),
]

# Download if not already available
for path, name in REQUIRED_NLTK_RESOURCES:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(name, download_dir=NLTK_PATH)

# Then proceed to your pipeline setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#cleaning + lemmatization function
def clean_text(text):
    # Lowercase
    text = text.lower()
    
    # Handle negations (combine with next word if possible)
    text = re.sub(r"\b(no|not|never)\s+(\w+)", r"no_\2", text)

    # Remove punctuation except underscores (used in negation)
    text = re.sub(r"[^\w\s_]", "", text)

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Lemmatize and remove stopwords (except 'no_x' preserved words)
    cleaned = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if (token in stop_words and token.startswith('no_')) or (token not in stop_words and len(token) > 2)
    ]

    return " ".join(cleaned)

def clean_texts(texts):
    return [clean_text(text) for text in texts]


# Models and pipeline loading
@st.cache_resource
def load_nlp_model():
    path = Path(__file__).parent / "models" / "nlp_pipeline.pkl"
    return joblib.load(path)

@st.cache_resource
def load_env_model():
    path = Path(__file__).parent / "models" / "environmental.pkl"
    return joblib.load(path)

nlp_pipeline = load_nlp_model() # NLP pipeline
model = load_env_model() # Environmental model

@st.cache_data
def load_main_data():
    path = Path(__file__).parent / "data" / "environmental.csv"
    return pd.read_csv(path)

df = load_main_data()


# Navigation options
st.sidebar.title("📍 CleanWatAI Navigation")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "NLP Page",
    "Quick Insights and Reports",
    "Water Point Contamination Risk Map",
    "Water Point Data Analysis"
])

# Route to selected page
if page == "Home":
    st.title("Welcome to CleanWatAI 💧")
    st.markdown("""
    CleanWatAI is a predictive analytics tool helping identify water quality risks in rural communities.
    
    Use the sidebar to:
    - Classify water reports (NLP)
    - View data insights and trends
    - Visualize contamination risk maps
    - Explore raw water point data
    """)

elif page == "NLP Page":
    st.title("🧠 NLP-Based Water Report Classification")
    st.markdown("Use the form below to classify water safety based on textual observations.")

    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        user_text = st.text_area(label="Describe what you want to know", height=150)

        col1, col2, col3 = st.columns(3)
        with col1:
            color = st.selectbox("Water Color", options=["", "Clear", "Brown", "Green", "Other"])
        with col2:
            clarity = st.selectbox("Clarity", options=["", "Clear", "Murky"])
        with col3:
            odor = st.selectbox("Odor", options=["", "None", "Chemical", "Sewage", "Other"])

        col1, col2, col3 = st.columns(3)
        with col1:
            rain = st.selectbox("Recent Rain", options=["", "No recent rain", "Light rain", "Heavy rain"])
        with col2:
            activity = st.selectbox("Nearby Activity", options=["", "Residential", "Industrial", "Agricultural", "None"])
        with col3:
            infrastructure = st.selectbox("Infrastructure", options=["", "Good condition", "Needs repair", "Unknown"])
        context_parts = []

        if color:
            context_parts.append(f"The water appears {color.lower()} in color.")
        if clarity:
            context_parts.append(f"It is {clarity.lower()} in clarity.")
        if odor:
            context_parts.append(f"It has a {odor.lower()} odor.")

        if rain:
            context_parts.append(f"There was {rain.lower()}.")
        if activity:
            context_parts.append(f"The area nearby is {activity.lower()}.")
        if infrastructure:
            context_parts.append(f"The infrastructure is (in) {infrastructure.lower()}.")

        # Combine original input + context
        combined_description = user_text.strip() + " " + " ".join(context_parts)

        # Edited text area with combined description from the select boxes and user input
        edited_description = st.text_area("📝 Final Input to the Model (Editable)", value=combined_description, height=200)

        col1, col2, col3 = st.columns(3)
        with col3:
            if st.button("Submit", type="primary", use_container_width=True):
                # Only runs when Submit is clicked
                if not edited_description or edited_description.strip() == "":
                    st.warning("Please describe your concern in the text area above.")
                else:
                    prediction = nlp_pipeline.predict([edited_description])[0]
                    probability = nlp_pipeline.predict_proba([edited_description])[0][prediction]

                    # Map prediction to label
                    label_map = {0: "Safe", 1: "Unsafe"}
                    prediction_label = label_map[prediction]

                    # Display result with appropriate style
                    if prediction_label == "Safe":
                        st.success(f"✅ Water is predicted to be SAFE.\nConfidence: {probability:.2%}")
                    else:
                        st.error(f"⚠️ Water is predicted to be UNSAFE.\nConfidence: {probability:.2%}")

    with st.container(border=True):
            st.caption("© 2025 CleanWaterAI. Data sourced from WPDx and other public datasets.")

elif page == "Quick Insights and Reports":
    st.title("📊 Quick Insights and Reports")
    st.markdown("Here is a quick snapshot of current water safety reports across regions.")

    col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1],gap="large")
    with col2:
        with st.container():
            st.subheader("Quick Insights")
            st.text("")
            
            #Load data
            df = df.dropna(subset=["latitude", "longitude"])
            df.dropna(axis=1, how="all", inplace=True)
            # Get unique locations
            unique_locations = df["clean_adm2"].dropna().unique()
            selected_location = st.selectbox("Select Location", sorted(unique_locations))

            # Filter by location
            filtered_df = df[df["clean_adm2"] == selected_location]

            # Risk score mapping
            risk_labels = {
                0: "🟢 Safe Quality",
                1: "🟡 Low Risk",
                2: "🟠 Medium Risk",
                3: "🔴 High Risk"
            }

            # Calculate quick stats
            num_stations = len(filtered_df)
            most_common_risk = filtered_df["risk_score"].mode().iloc[0]
            trend = "↗️ Slight increase"  # Optional: Replace with real trend logic later

            # Risk score summary
            st.subheader(f"{selected_location} Risk Score Summary")
            st.text(f"Monitoring Stations: {num_stations} active")
            st.text("")
            risk_counts_raw = filtered_df["risk_score"].value_counts().sort_index()
            for score, count in risk_counts_raw.items():
                label = risk_labels.get(score, f"Risk {score}")
                st.text(f"{label}: {count}")

            # Sort and drop zero-counts for pie chart
            risk_counts = filtered_df["risk_score"].value_counts().reindex([0, 1, 2, 3], fill_value=0)
            risk_counts = risk_counts[risk_counts > 0]  # Only keep non-zero risk levels

            # Prepare labels and colors (only for present scores)
            labels = [risk_labels.get(score) for score in risk_counts.index]
            color_map = {0: 'green', 1: 'gold', 2: 'orange', 3: 'red'}
            colors = [color_map[score] for score in risk_counts.index]

            # Create pie chart
            fig, ax = plt.subplots()
            ax.pie(risk_counts, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
            ax.axis("equal")  # Make the pie chart circular

            st.pyplot(fig)

    with col3:
        with st.container():
            st.subheader("Latest Alerts")
            st.text("")
            st.text("🦠 Microbial contamination")
            st.text("Kiambiu area • 2h ago")
            st.text("")
            st.text("🌊 High turbidity levels")
            st.text("Industrial zone • 3h ago")
            st.text("")
            st.text("⚗️ Chemical levels elevated")
            st.text("Industrial zone • 1d ago")
            st.text("")
            st.button("View All", key="view_alerts")
    with col4:
        with st.container():
            st.subheader("Reports & Analytics")
            st.text("")
            st.text("📈 Water Quality Trends")
            st.text("Urban Areas • Jul 15")
            st.text("")
            st.text("🌍 Regional Analysis")
            st.text("East Africa • Jun 30")
            st.text("")
            st.text("🏙️ Infrastructure Assessment")
            st.text("CBD Systems • Jun 22")
            st.text("")
            st.button("View All", key="view_reports")

    with st.container(border=True):
            st.caption("© 2025 CleanWaterAI. Data sourced from WPDx and other public datasets.")

elif page == "Water Point Contamination Risk Map":
    st.title("📍 Water Point Contamination Risk Map")
    st.markdown("This map shows predicted contamination risk for water points based on environmental features.")

    map_cols = st.columns([.5, 10, .5])
    with map_cols[1]:
        with st.container(border=True):
            st.text("")
            
        #Load data
        df = df.dropna(subset=["latitude", "longitude"])
        df.dropna(axis=1, how="all", inplace=True)

        #Add location name
        df["location_name"] = (
            df["clean_adm3"].fillna("") + ", " +
            df["clean_adm2"].fillna("") + ", " +
            df["clean_adm1"].fillna("")
        )

        # Predict risk
        df["predicted_risk"] = model.predict(df)
        
        def risk_label(r):
            return {
                0: "🟢 Safe Quality",
                1: "🟡 Low Risk",
                2: "🟠 Medium Risk",
                3: "🔴 High Risk"
            }.get(r, "Unknown")

        def risk_color(r):
            return {
                0: [0, 255, 0, 160],
                1: [255, 255, 0, 160],
                2: [255, 165, 0, 160],
                3: [255, 0, 0, 160]
            }.get(r, [128, 128, 128, 160])

        #print("📦 Available columns in df:", df.columns.tolist())
        #print(df.head())

        df["risk_label"] = df["predicted_risk"].apply(risk_label)
        df["color"] = df["predicted_risk"].apply(risk_color)
        df["risk_label_clean"] = df["risk_label"].replace({
            "🔴 High Risk": "High Risk",
            "🟠 Medium Risk": "Medium Risk",
            "🟡 Low Risk": "Low Risk",
            "🟢 Safe Quality": "Safe Quality"
        })

        df["quality_score"] = (1 - df["predicted_risk"] / 3 * 0.75) * 100
        df["quality_score"] = df["quality_score"].round(1)

        df["risk_level"] = df["risk_label_clean"]

        if "contamination_type" not in df.columns:
            df["contamination_type"] = "Not available"


        available_risks = sorted(df["risk_label"].unique().tolist())
        selected_risks = st.multiselect("Filter by risk level:", available_risks, default=available_risks)
        filtered_df = df[df["risk_label"].isin(selected_risks)]

        # Handle case where no points are selected — show blank map
        if filtered_df.empty:
            st.info("No risk levels selected. Displaying a blank global map.")

            # Set global view
            view_state = pdk.ViewState(
                latitude=0.0,
                longitude=0.0,
                zoom=1,
                pitch=0
            )

            # Set empty DataFrame for rendering
            filtered_df = pd.DataFrame(columns=["latitude", "longitude", "location_name", "color", "risk_label"])
        else:
            view_state = pdk.ViewState(
                latitude=filtered_df["latitude"].mean(),
                longitude=filtered_df["longitude"].mean(),
                zoom=6,
                pitch=0
            )

        # Build map layer
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=filtered_df,
            get_position='[longitude, latitude]' if not filtered_df.empty else None,
            get_fill_color="color" if not filtered_df.empty else None,
            get_radius=3000,
            pickable=True,
        )

        # Tooltip definition
        tooltip = {
            "html": "<b>{location_name}</b><br/>"
                    "Risk: {risk_label}<br/>"
                    "Lat: {latitude}<br/>Lon: {longitude}",
            "style": {"backgroundColor": "white", "color": "black"}
        }

        # Show map
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip
        ))

        # Point selector and detail box
        if not filtered_df.empty:
            selected_point = st.selectbox(
                "Select a monitoring point for details:",
                options=filtered_df['location_name'].tolist()
            )

            if selected_point:
                point_data = filtered_df[filtered_df['location_name'] == selected_point].iloc[0]

                with st.container(border=True):
                    st.subheader("Point Details")
                    st.text("")

                    detail_col1, detail_col2 = st.columns(2)

                    with detail_col1:
                        st.metric("Quality Score", f"{point_data.get('quality_score', 'N/A')}/100")
                        st.text(f"Risk Level: {point_data.get('risk_level', 'Unknown')}")
                        st.text(f"Local Population: {point_data.get('local_population', 'N/A')}")

                    with detail_col2:
                        st.text(f"Contamination: {point_data.get('contamination_type', 'Not available')}")
                        st.text(f"Location: {point_data['latitude']:.4f}, {point_data['longitude']:.4f}")
                        st.text(f"Served Population: {point_data.get('served_population', 'N/A')}")

                    st.text("")
                    st.text("")
                    st.text("")

            
            # Add bottom padding
            st.text("")
    
    st.text("")  # Vertical spacing
    st.text("")  # Additional vertical spacing
    
    monitoring_header_cols = st.columns([.5, 10, .5])
    with monitoring_header_cols[1]:
        st.subheader("Monitoring Data")
    
    monitoring_outer_cols = st.columns([.5, 10, .5])
    with monitoring_outer_cols[1]:
        with st.container(border=True):
            st.text("")
            tab1, tab2 = st.tabs(["Coverage Statistics", "Data Sources"])
            
            with tab1:
                total_points = len(filtered_df)
                st.metric("Total Points", total_points)
                st.text("Coverage: 95% of region") 
                st.text("Updated: Today")
            
            with tab2:
                st.text("Sources:")
                st.text("• WPDx (Water Point Data Exchange)")
                st.text("• Google Earth Engine")
                st.text("• Field Observations")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
            st.caption("© 2025 CleanWaterAI. Data sourced from WPDx and other public datasets.")

elif page == "Water Point Data Analysis":
    st.title("🔬 Data Analysis")
    st.write("Explore and analyze water point datasets.")

    data_cols = st.columns([.5, 10, .5])
    with data_cols[1]:
        st.text("Showing the most recent water point data with model-predicted risk levels")
        st.text("Last updated: August 3, 2025")
        st.text("")

        with st.container(border=True):
            data_tab1, data_tab2, data_tab3, data_tab4 = st.tabs(["All Data", "Functional Status", "Risk Analysis", "Quality Trend"])
            
            # TAB 1: ALL DATA
            with data_tab1:
                st.dataframe(
                    df,
                    column_config={
                        "water_point_id": "Water Point ID",
                        "country_name": "Country",
                        "region": "Region",
                        "district": "District",
                        "water_source": "Water Source",
                        "status": st.column_config.SelectboxColumn(
                            "Status",
                            help="Operational status of the water point",
                            width="medium",
                            options=[
                                "Functional",
                                "Non-functional",
                                "Needs repair"
                            ]
                        ),
                        "date_installed": st.column_config.DateColumn(
                            "Installation Date",
                            format="MMM DD, YYYY",
                        ),
                        "latest_record": st.column_config.DateColumn(
                            "Last Updated",
                            format="MMM DD, YYYY",
                        ),
                        "water_quality": "Water Quality",
                        "risk_score": st.column_config.ProgressColumn(
                            "Risk Score",
                            help="Higher score = lower risk",
                            format="%d",
                            min_value=0,
                            max_value=100,
                        ),
                        "risk_level": "Risk Level"
                    },
                    hide_index=True,
                    use_container_width=True
                )

            # TAB 2: FUNCTIONAL STATUS
            def clean_status(status):
                status = status.strip().lower()

                if "non-functional" in status:
                    return "Non-Functional"
                elif "functional" in status:
                    return "Functional"
                elif "abandoned" in status or "decommissioned" in status:
                    return "Decommissioned"
                else:
                    return "Unknown"

            df['status_clean'] = df['status_clean'].fillna("Unknown").apply(clean_status)

            with data_tab2:
                status_counts = df['status_clean'].value_counts().reset_index()
                status_counts.columns = ['Status', 'Count']

                
                st.bar_chart(status_counts, x='Status', y='Count')
                
                st.text("Filter data by status:")
                status_filter = st.multiselect(
                    "Select status to view:",
                    options=df['status_clean'].unique(),
                    default=df['status_clean'].unique()
                )
                
                filtered_status_data = df[df['status_clean'].isin(status_filter)]
                st.dataframe(filtered_status_data, hide_index=True, use_container_width=True)

            # TAB 3: RISK ANALYSIS
            with data_tab3:
                df["risk_label"] = df["predicted_risk"].apply(risk_label)
                df["color"] = df["predicted_risk"].apply(risk_color)
                df["risk_label_clean"] = df["risk_label"].replace({
                    "🔴 High Risk": "High Risk",
                    "🟠 Medium Risk": "Medium Risk",
                    "🟡 Low Risk": "Low Risk",
                    "🟢 Safe Quality": "Safe Quality"
                })

                df["quality_score"] = (1 - df["predicted_risk"] / 3 * 0.75) * 100
                df["quality_score"] = df["quality_score"].round(1)

                df["risk_level"] = df["risk_label_clean"]
                risk_counts = df['risk_level'].value_counts().reset_index()
                risk_counts.columns = ['Risk Level', 'Count']

                st.text("Average Risk Score by Region")
                region_risk = df.groupby('clean_adm1')['risk_score'].mean().reset_index()
                region_risk.columns = ['Region', 'Average Risk Score']

                st.bar_chart(region_risk, x='Region', y='Average Risk Score')



                st.text("Filter data by risk level:")
                risk_filter = st.multiselect(
                    "Select risk level to view:",
                    options=df['risk_level'].unique(),
                    default=df['risk_level'].unique()
                )

                filtered_risk_data = df[df['risk_level'].isin(risk_filter)]
                st.dataframe(filtered_risk_data, hide_index=True, use_container_width=True)

            # TAB 4: QUALITY TREND
            with data_tab4:
                st.text("Water Quality Trend Over Time")
                st.text("")

                # ✅ Step 1: Define the new column (do this outside any widget)
                df['point_id'] = df['location_name'].fillna('Unnamed') + ' (' + df['latitude'].astype(str) + ', ' + df['longitude'].astype(str) + ')'

                # ✅ Step 2: Use it in the selectbox
                selected_water_point = st.selectbox(
                    "Select a water point to view quality trend:",
                    options=df['point_id'].tolist()
                )


                if selected_water_point:
                    point_data = df[df['point_id'] == selected_water_point].iloc[0]

                    st.text(f"30-day quality trend for {selected_water_point}")
                    st.text("")

                    days = 30
                    dates = pd.date_range(end=pd.Timestamp.now(), periods=days)

                    base_quality = point_data['risk_score']
                    noise = np.random.normal(0, 5, days)
                    trend = np.linspace(-10, 10, days)

                    quality_trend = np.clip(base_quality + trend + noise, 0, 100)

                    trend_data = pd.DataFrame({
                        'date': dates,
                        'quality': quality_trend
                    })

                    st.line_chart(trend_data.set_index('date'))

                    avg_quality = quality_trend.mean()
                    st.text(f"Average Quality Score: {avg_quality:.1f}/100")

                    if trend[-1] > trend[0]:
                        st.text("Trend: ↗️ Improving")
                    elif trend[-1] < trend[0]:
                        st.text("Trend: ↘️ Declining")
                    else:
                        st.text("Trend: → Stable")

            # CSV download
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Full Water Data (CSV)",
                data=csv,
                file_name="cleanwat_ai_water_data.csv",
                mime="text/csv"
            )

            st.text("")
            st.text("")
            with st.container(border=True):
                st.caption("© 2025 CleanWaterAI. Data sourced from WPDx and other public datasets.")
