import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Dashboard Analisis PHK Indonesia",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark/light mode
def load_custom_css(theme):
    if theme == "Dark":
        st.markdown("""
        <style>
        .main {
            background-color: #0e1117;
            color: #fafafa;
        }
        .stMetric {
            background-color: #262730;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 15px;
            color: white;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
            margin: 10px 0;
        }
        .section-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            color: white;
            text-align: center;
        }
        h1, h2, h3 {
            color: #fafafa;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #262730;
            border-radius: 10px 10px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #667eea;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .main {
            background-color: #ffffff;
            color: #262730;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 15px;
            color: white;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            margin: 10px 0;
        }
        .section-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            color: white;
            text-align: center;
        }
        h1, h2, h3 {
            color: #262730;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 10px 10px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #667eea;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():

    np.random.seed(42)
    n_samples = 1000
    
    data = pd.read_csv('phk_indo_fix.csv')
    
    return data

# Initialize session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'
if 'data' not in st.session_state:
    st.session_state.data = load_data()

# Load data
df = st.session_state.data

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/business-report.png", width=80)
    st.title("🎛️ Control Panel")
    
    # Theme toggle
    st.markdown("---")
    theme_col1, theme_col2 = st.columns([3, 1])
    with theme_col1:
        st.write("**Display Mode:**")
    with theme_col2:
        if st.button("🌓"):
            st.session_state.theme = 'Dark' if st.session_state.theme == 'Light' else 'Light'
            st.rerun()
    
    st.info(f"Mode: **{st.session_state.theme}**")
    
    st.markdown("---")
    st.subheader("📅 Data Filters")
    
    # Year filter
    years = sorted(df['tahun'].unique())
    selected_years = st.multiselect(
        "Select Years",
        options=years,
        default=years
    )
    
    # Province filter
    provinces = sorted(df['provinsi'].unique())
    selected_provinces = st.multiselect(
        "Select Provinces",
        options=provinces,
        default=provinces
    )
    
    # Sector filter
    sectors = sorted(df['sektor'].unique())
    selected_sectors = st.multiselect(
        "Select Sectors",
        options=sectors,
        default=sectors
    )
    
    # Scale filter
    scales = sorted(df['skala_phk'].unique())
    selected_scales = st.multiselect(
        "Select PHK Scale",
        options=scales,
        default=scales
    )
    
    # Apply custom CSS
    load_custom_css(st.session_state.theme)

    # Filter data ← INI HARUS DULUAN
    filtered_df = df[
        (df['tahun'].isin(selected_years)) &
        (df['provinsi'].isin(selected_provinces)) &
        (df['sektor'].isin(selected_sectors)) &
        (df['skala_phk'].isin(selected_scales))
    ]

    # BARU SEKARANG download button di sidebar
    st.markdown("---")
    st.subheader("📥 Download Data")

    @st.cache_data
    def convert_df_to_csv(dataframe):
        return dataframe.to_csv(index=False).encode('utf-8')

    csv_data = convert_df_to_csv(filtered_df)  # ✅ SEKARANG filtered_df sudah ada
    st.download_button(
        label="📄 Download CSV Filtered Data",
        data=csv_data,
        file_name=f"phk_data_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("📊 About")
    st.info("""
    Dashboard ini menampilkan analisis komprehensif 
    data PHK di Indonesia menggunakan teknik 
    Clustering dan Regresi Linear.
    """)



# Main content
st.title("📊 Dashboard Analisis PHK Indonesia")
st.markdown("### Analisis Komprehensif Menggunakan Clustering dan Regresi Linear")

st.markdown("---")

# Key Metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_phk = filtered_df['total_phk'].sum()
    st.metric("Total PHK", f"{total_phk:,.0f}")

with col2:
    avg_phk = filtered_df['total_phk'].mean()
    st.metric("Rata-rata PHK", f"{avg_phk:,.0f}")

with col3:
    total_events = len(filtered_df)
    st.metric("Jumlah Kejadian", f"{total_events:,}")

with col4:
    total_provinces = filtered_df['provinsi'].nunique()
    st.metric("Jumlah Provinsi", f"{total_provinces}")

with col5:
    total_sectors = filtered_df['sektor'].nunique()
    st.metric("Jumlah Sektor", f"{total_sectors}")

st.markdown("---")

# Tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Overview", 
    "🔍 Clustering Analysis",
    "📈 Regression Analysis",
    "📊 Data Explorer"
])

# TAB 1: Overview
with tab1:
    st.header("📈 Data Overview & Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Trend over years
        yearly_trend = filtered_df.groupby('tahun')['total_phk'].sum().reset_index()
        fig_trend = px.area(
            yearly_trend,
            x='tahun',
            y='total_phk',
            title='Tren PHK per Tahun',
            labels={'tahun': 'Tahun', 'total_phk': 'Total PHK'},
            color_discrete_sequence=['#667eea']
        )
        fig_trend.update_layout(height=400)
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        # Distribution by scale
        scale_dist = filtered_df['skala_phk'].value_counts().reset_index()
        scale_dist.columns = ['skala_phk', 'count']
        
        fig_scale = px.pie(
            scale_dist,
            values='count',
            names='skala_phk',
            title='Distribusi Skala PHK',
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig_scale.update_layout(height=400)
        st.plotly_chart(fig_scale, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Top provinces
        top_provinces = filtered_df.groupby('provinsi')['total_phk'].sum().nlargest(10).reset_index()
        fig_provinces = px.bar(
            top_provinces,
            y='provinsi',
            x='total_phk',
            orientation='h',
            title='Top 10 Provinsi dengan PHK Tertinggi',
            labels={'provinsi': 'Provinsi', 'total_phk': 'Total PHK'},
            color='total_phk',
            color_continuous_scale='Viridis'
        )
        fig_provinces.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_provinces, use_container_width=True)
    
    with col4:
        # Top sectors
        top_sectors = filtered_df.groupby('sektor')['total_phk'].sum().nlargest(10).reset_index()
        fig_sectors = px.bar(
            top_sectors,
            y='sektor',
            x='total_phk',
            orientation='h',
            title='Top 10 Sektor dengan PHK Tertinggi',
            labels={'sektor': 'Sektor', 'total_phk': 'Total PHK'},
            color='total_phk',
            color_continuous_scale='Plasma'
        )
        fig_sectors.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_sectors, use_container_width=True)
    
    # Heatmap: Province vs Year
    st.subheader("🌡️ Heatmap: Provinsi vs Tahun")
    heatmap_data = filtered_df.pivot_table(
        values='total_phk',
        index='provinsi',
        columns='tahun',
        aggfunc='sum',
        fill_value=0
    )
    
    fig_heatmap = px.imshow(
        heatmap_data,
        text_auto='.0f',
        title='Total PHK per Provinsi per Tahun',
        color_continuous_scale='YlOrRd',
        aspect='auto'
    )
    fig_heatmap.update_layout(height=600)
    st.plotly_chart(fig_heatmap, use_container_width=True)

# TAB 2: Clustering Analysis
with tab2:
    st.header("🔍 Clustering Analysis (K-Means)")
    
    # Prepare data for clustering
    clustering_data = filtered_df.groupby('provinsi').agg({
        'total_phk': ['sum', 'mean', 'count'],
        'sektor': 'nunique',
        'skala_phk': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
    }).reset_index()
    
    clustering_data.columns = ['provinsi', 'total_phk_sum', 'avg_phk', 'phk_frequency', 'sector_diversity', 'dominant_scale']
    
    # Encode dominant scale
    le_scale = LabelEncoder()
    clustering_data['scale_encoded'] = le_scale.fit_transform(clustering_data['dominant_scale'])
    
    # Features for clustering
    X_cluster = clustering_data[['avg_phk', 'phk_frequency', 'sector_diversity', 'scale_encoded']]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clustering_data['cluster'] = kmeans.fit_predict(X_scaled)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        clustering_data['pca1'] = X_pca[:, 0]
        clustering_data['pca2'] = X_pca[:, 1]
        
        fig_cluster = px.scatter(
            clustering_data,
            x='pca1',
            y='pca2',
            color='cluster',
            hover_data=['provinsi', 'avg_phk', 'phk_frequency'],
            title=f'Clustering Provinsi (K-Means, k={2})',
            labels={'pca1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', 
                   'pca2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'},
            color_continuous_scale='Viridis'
        )
        fig_cluster.update_layout(height=500)
        st.plotly_chart(fig_cluster, use_container_width=True)
    
    with col2:
        # Cluster characteristics
        cluster_summary = clustering_data.groupby('cluster').agg({
            'avg_phk': 'mean',
            'phk_frequency': 'mean',
            'sector_diversity': 'mean',
            'provinsi': 'count'
        }).round(2)
        
        cluster_summary.columns = ['Avg PHK', 'Frequency', 'Sector Diversity', 'Num Provinces']
        st.subheader("📊 Cluster Characteristics")
        st.dataframe(cluster_summary, use_container_width=True)
        
        # Provinces per cluster
        st.subheader("🏛️ Provinces per Cluster")
        for cluster_id in sorted(clustering_data['cluster'].unique()):
            cluster_provinces = clustering_data[clustering_data['cluster'] == cluster_id]['provinsi'].tolist()
            with st.expander(f"Cluster {cluster_id} ({len(cluster_provinces)} provinces)"):
                st.write(", ".join(cluster_provinces))
    
    # Feature importance in clustering
    st.subheader("🎯 Feature Importance in Clustering")
    
    # Calculate feature importance based on cluster means
    feature_importance = pd.DataFrame({
        'Feature': ['Avg PHK', 'Frequency', 'Sector Diversity', 'Scale'],
        'Importance': np.std(kmeans.cluster_centers_, axis=0)
    }).sort_values('Importance', ascending=False)
    
    fig_importance = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance in Clustering',
        labels={'Importance': 'Standard Deviation across Clusters', 'Feature': 'Feature'},
        color='Importance',
        color_continuous_scale='Blues'
    )
    fig_importance.update_layout(height=400)
    st.plotly_chart(fig_importance, use_container_width=True)

# TAB 3: Regression Analysis
with tab3:
    st.header("📈 Regression Analysis")
    
    # Multiple Linear Regression
    st.subheader("🔮 Multiple Linear Regression")
    
    # Prepare data for regression
    regression_data = filtered_df.groupby(['tahun', 'provinsi', 'sektor']).agg({
        'total_phk': 'sum',
        'skala_phk': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
    }).reset_index()
    
    # Encode categorical variables
    le_prov = LabelEncoder()
    le_sektor = LabelEncoder()
    le_skala = LabelEncoder()
    
    regression_data['provinsi_encoded'] = le_prov.fit_transform(regression_data['provinsi'])
    regression_data['sektor_encoded'] = le_sektor.fit_transform(regression_data['sektor'])
    regression_data['skala_encoded'] = le_skala.fit_transform(regression_data['skala_phk'])
    
    # Features and target
    X_reg = regression_data[['tahun', 'provinsi_encoded', 'sektor_encoded', 'skala_encoded']]
    y_reg = regression_data['total_phk']
    
    if len(X_reg) > 10:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", f"{r2:.3f}")
        with col2:
            st.metric("RMSE", f"{rmse:.0f}")
        with col3:
            st.metric("MAE", f"{mae:.0f}")
        with col4:
            st.metric("Samples", f"{len(X_reg)}")
        
        # Feature coefficients
        st.subheader("📊 Feature Coefficients")
        coefficients = pd.DataFrame({
            'Feature': ['Tahun', 'Provinsi', 'Sektor', 'Skala PHK'],
            'Coefficient': model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        fig_coeff = px.bar(
            coefficients,
            x='Coefficient',
            y='Feature',
            orientation='h',
            title='Feature Coefficients in Linear Regression',
            color='Coefficient',
            color_continuous_scale='RdYlGn',
            labels={'Coefficient': 'Coefficient Value', 'Feature': 'Feature'}
        )
        fig_coeff.update_layout(height=400)
        st.plotly_chart(fig_coeff, use_container_width=True)
        
        # Actual vs Predicted
        st.subheader("📈 Actual vs Predicted")
        comparison_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })
        
        fig_comparison = px.scatter(
            comparison_df,
            x='Actual',
            y='Predicted',
            title='Actual vs Predicted Values',
            labels={'Actual': 'Actual PHK', 'Predicted': 'Predicted PHK'},
            trendline="lowess"
        )
        fig_comparison.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        fig_comparison.update_layout(height=500)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Prediction tool
        st.subheader("🔮 Prediction Tool")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            input_year = st.number_input("Tahun", min_value=2020, max_value=2030, value=2024)
        with col2:
            input_province = st.selectbox("Provinsi", options=le_prov.classes_)
        with col3:
            input_sector = st.selectbox("Sektor", options=le_sektor.classes_)
        with col4:
            input_scale = st.selectbox("Skala PHK", options=le_skala.classes_)
        
        if st.button("Predict", type="primary"):
            province_encoded = le_prov.transform([input_province])[0]
            sector_encoded = le_sektor.transform([input_sector])[0]
            scale_encoded = le_skala.transform([input_scale])[0]
            
            input_data = [[input_year, province_encoded, sector_encoded, scale_encoded]]
            prediction = model.predict(input_data)[0]
            
            st.success(f"### Predicted PHK: **{prediction:,.0f}** orang")
    
    else:
        st.warning("Not enough data for regression analysis. Please adjust filters.")

# TAB 4: Data Explorer
with tab4:
    st.header("📊 Data Explorer")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📋 Filtered Data")
        st.dataframe(filtered_df, use_container_width=True)
    
    with col2:
        st.subheader("📥 Export Data")
        
        # CSV download
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"phk_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown("---")
        st.subheader("📈 Statistics")
        st.write(f"**Total Rows:** {len(filtered_df):,}")
        st.write(f"**Total PHK:** {filtered_df['total_phk'].sum():,}")
        st.write(f"**Average PHK:** {filtered_df['total_phk'].mean():.0f}")
        st.write(f"**Max PHK:** {filtered_df['total_phk'].max():,}")
        st.write(f"**Min PHK:** {filtered_df['total_phk'].min():,}")
    
    # Correlation heatmap
    st.subheader("🔗 Correlation Analysis")
    
    # Prepare numeric data for correlation
    numeric_data = filtered_df.copy()
    numeric_data['tahun_numeric'] = numeric_data['tahun']
    numeric_data['provinsi_numeric'] = le_prov.fit_transform(numeric_data['provinsi'])
    numeric_data['sektor_numeric'] = le_sektor.fit_transform(numeric_data['sektor'])
    numeric_data['skala_numeric'] = le_skala.fit_transform(numeric_data['skala_phk'])
    
    corr_data = numeric_data[['total_phk', 'tahun_numeric', 'provinsi_numeric', 'sektor_numeric', 'skala_numeric']].corr()
    
    fig_corr = px.imshow(
        corr_data,
        text_auto='.2f',
        title='Correlation Matrix',
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>📊 Dashboard Analisis PHK Indonesia - Clustering & Regression Analysis</p>
    <p>Dibuat dengan ❤️ menggunakan Streamlit & Plotly</p>
</div>
""", unsafe_allow_html=True)