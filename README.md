# CleanWater AI 💧

## Project Overview

CleanWater AI is an end-to-end machine learning system for monitoring and predicting water quality in Kenya. The system integrates multiple data sources, applies machine learning models, and provides real-time insights through an interactive dashboard.

### Business Understanding
**Problem**: Access to clean water remains a critical challenge in Kenya, with limited real-time monitoring and predictive capabilities for water quality assessment.

**Solution**: An AI-powered system that:
- Monitors water point status and quality indicators
- Predicts contamination risks using environmental factors
- Provides real-time alerts and interactive visualizations
- Enables citizen reporting and community engagement

### Data Sources & Integration

**Primary Data Sources:**
1. **WPDx (Water Point Data Exchange)**: 22,000+ water points across Kenya with location, status, and infrastructure data
2. **Google Earth Engine**: Environmental satellite data including NDVI, rainfall, temperature, soil moisture
3. **User Input**: Citizen reports on water quality observations (color, clarity, odor, infrastructure)

**Output Formats:**
- **Interactive Dashboard**: Streamlit web application with maps, charts, and alerts
- **Real-time Predictions**: ML model predictions on water quality risks
- **Alert System**: Automated notifications for contamination risks
- **CSV Exports**: Downloadable datasets for further analysis

## Project Architecture & Data Flow

```
Data Extraction → Data Processing → Model Training → Deployment → Monitoring
     ↓                ↓               ↓            ↓           ↓
   WPDx API      →  Cleaning     →  XGBoost    →  Streamlit  →  Alerts
   GEE API       →  Merging      →  NLP Model  →  Live API   →  Updates
   User Input    →  Feature Eng  →  Evaluation →  Dashboard  →  Feedback
```

### Folder Structure

```
cleanwaterai/
├── 📁 app/                                       # Streamlit application & live services
│   ├── streamlit_app.py                          # Main dashboard interface
│   ├── trigger_ingestion.py                      # Live data ingestion scheduler
│   ├── trigger_predictions.py                    # Real-time prediction engine
│   ├── trigger_alerts.py                         # Alert notification system
│   └── index.css                                 # Dashboard styling
│
├── 📁 data/                                      # Data storage hierarchy
│   ├── raw/                                      # Original source data
│   │   ├── wpdx_kenya.csv                        # Water Point Data Exchange (22K points)
│   │   └── ndvi_scaled.csv                       # Satellite environmental data
│   └── processed/                                # Cleaned, merged datasets
│       └── final_wpdx_environmental_data.csv     # ML-ready dataset
│
├── 📁 scripts/                                   # Core data pipeline scripts
│   ├── extract_data.py                           # WPDx API data extraction
│   ├── merge_data.py                             # Data integration & joining
│   ├── prepare_data.py                           # Feature engineering pipeline
│   ├── train_xgb.py                              # XGBoost model training
│   ├── train_nlp.py                              # NLP model for text analysis
│   ├── ingest_live_wpdx.py                       # Live WPDx updates
│   ├── ingest_live_gee.py                        # Live satellite data
│   ├── ingest_live_nlp.py                        # Live text processing
│   └── deploy.py                                 # Model deployment utilities
│
├── 📁 notebooks/                                 # Jupyter analysis notebooks (CRISP-DM flow)
│   ├── 01_data_extraction.ipynb                  # Data Understanding
│   ├── 02_data_cleaning.ipynb                    # Data Preparation
│   ├── 03_data_merging.ipynb                     # Data Integration
│   ├── 04_data_preparation.ipynb                 # Feature Engineering
│   ├── 05_xgboost_model_training.ipynb           # Modeling
│   ├── 06_nlp_model_training.ipynb               # Text Analytics
│   ├── 07_model_evaluation.ipynb                 # Model Assessment
│   ├── 08_model_interpretability.ipynb           # Model Insights
│   ├── 09_deployment.ipynb                       # Deployment Strategy
│   └── 10_monitoring_and_maintenance.ipynb       # Production Monitoring
│
├── 📁 models/                                   # Trained model artifacts
│   └── nlp.pkl                                   # NLP model for text classification
│
├── 📁 reports/                                  # Documentation & presentations
│   ├── prd.pdf                                   # Product Requirements Document
│   ├── presentation.pdf                          # Project presentation
│   └── report.pdf                                # Technical report
│
├── main.py                                       # Main orchestration starting point
├── requirements.txt                              # Python dependencies
├── pyproject.toml                                # Build configuration
└── setup.py                                      # Package installation
```

### Data Flow Architecture

**1. Data Ingestion Pipeline:**
```
WPDx API → extract_data.py → data/raw/wpdx_kenya.csv
GEE API → satellite processing → data/raw/ndvi_scaled.csv
User Input → streamlit_app.py → real-time processing
```

**2. Data Processing Pipeline:**
```
Raw Data → merge_data.py → Data Integration
Integrated Data → prepare_data.py → Feature Engineering
Features → data/processed/final_wpdx_environmental_data.csv
```

**3. Model Training Pipeline:**
```
Processed Data → train_xgb.py → XGBoost Model (water quality prediction)
Text Data → train_nlp.py → NLP Model (text classification)
Models → Evaluation → models/ directory
```

**4. Live System Pipeline:**
```
trigger_ingestion.py → Live data updates every hour
trigger_predictions.py → Real-time ML predictions
trigger_alerts.py → Automated risk notifications
streamlit_app.py → Interactive dashboard display
```

## Quick Start Guide

### Prerequisites
- **Python Version**: 3.8+ (tested on 3.12)
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM recommended

### Installation & Setup

1. **Clone and Navigate to Project:**
```bash
git clone https://github.com/TonnieD/CleanWatAI.git
cd cleanwatai
```

2. Create a pyproject.toml file with the following content:
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"
```

3. **Install Package in Development Mode:**
```bash
pip install -e .
```

4. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

### Running the Complete Pipeline

**Option A: Full Automated Pipeline**
```bash
python main.py
```
This will:
- Execute all notebooks sequentially (01-10)
- Run all data scripts (extract → merge → prepare → train)
- Start live ingestion, predictions, and alerts
- Launch Streamlit dashboard on port 8505

**Option B: Manual Step-by-Step Execution**

1. **Data Extraction:**
```bash
python scripts/extract_data.py
```

2. **Data Processing:**
```bash
python scripts/merge_data.py
python scripts/prepare_data.py
```

3. **Model Training:**
```bash
python scripts/train_xgb.py
python scripts/train_nlp.py
```

4. **Start Live Services:**
```bash
python app/trigger_ingestion.py
python app/trigger_predictions.py
python app/trigger_alerts.py
```

5. **Launch Dashboard:**
```bash
streamlit run app/streamlit_app.py --server.port 8505
```

### Working with Notebooks

**Sequential Analysis (CRISP-DM methodology):**
```bash
jupyter notebook notebooks/01_data_extraction.ipynb
# Continue through 02-10 in order
```

Each notebook builds on the previous one:
- **01-04**: Data understanding and preparation
- **05-06**: Model development and training  
- **07-08**: Model evaluation and interpretation
- **09-10**: Deployment and monitoring strategies

### Dashboard Features

The Streamlit application provides:

**Interactive Components:**
- 🗺️ **Risk Map**: Geospatial water quality visualization
- 📊 **Analytics Dashboard**: Real-time metrics and trends
- 🚨 **Alert System**: Live contamination warnings
- 📝 **Citizen Reporting**: Community water quality input
- 📈 **Trend Analysis**: Historical data visualization

**Real-time Data:**
- Live water point status updates
- Environmental satellite data integration
- ML-powered risk predictions
- Automated alert generation

### Development Workflow

**For New Contributors:**

1. **Environment Setup**: Follow installation steps above
2. **Data Pipeline**: Run `python main.py` to ensure full pipeline works
3. **Notebook Analysis**: Review notebooks 01-10 to understand methodology
4. **Feature Development**: Modify individual scripts in `scripts/` directory
5. **Dashboard Updates**: Edit `app/streamlit_app.py` for UI changes
6. **Testing**: Verify changes don't break the main pipeline

**Key Integration Points:**
- `scripts/extract_data.py` ↔ External APIs (WPDx, Google Earth Engine)
- `data/processed/` ↔ ML models for training and prediction
- `app/trigger_*.py` ↔ Live data ingestion and alert systems
- `app/streamlit_app.py` ↔ User interface and visualization

### Configuration Notes

**Environment Variables** (if needed):
- Google Earth Engine authentication
- Google CloudAPI keys for external services
- Database connection strings (for production)

**Performance Considerations:**
- Initial data extraction may take 15-20 minutes (22K records)
- Model training requires ~2GB memory for full dataset
- Dashboard loads data from CSV files (consider database for production)

### Troubleshooting

**Common Issues:**
1. **Memory errors**: Reduce batch size in data processing scripts
2. **API timeouts**: Check internet connection and API rate limits
3. **Missing models**: Ensure `scripts/train_*.py` completed successfully
4. **Dashboard errors**: Verify all CSV files exist in `data/` directories

**Support**: Check individual script documentation and notebook markdown cells for detailed implementation notes.
