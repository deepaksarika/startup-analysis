import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from wordcloud import WordCloud
from textblob import TextBlob
import altair as alt
from datetime import datetime
import seaborn as sns
from PIL import Image
import re
import folium
from streamlit_folium import folium_static
from streamlit_option_menu import option_menu
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
st.set_page_config(
    page_title="Startup Insights Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f0f2f6;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0277BD;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #616161;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# App config


# Load dataset
@st.cache_data
def load_data():
    file_path = "startups_cleaned.xlsx"
    df = pd.read_excel(file_path, sheet_name='startups_with_key_value_columns')
    
    # Clean Funding Column
    df['Funding'] = df['Funding'].astype(str).str.replace('[^\d.]', '', regex=True)
    df['Funding'] = pd.to_numeric(df['Funding'], errors='coerce')
    
    # Drop NaN values in Funding
    df = df.dropna(subset=['Funding'])
    
    # Data Preprocessing
    df['Started in'] = pd.to_numeric(df['Started in'], errors='coerce')
    
    # Calculate age of startups
    current_year = datetime.now().year
    df['Age'] = current_year - df['Started in']
    
    df['Number of employees'] = pd.to_numeric(df['Number of employees'], errors='coerce')
    df['Funding per Employee'] = df['Funding'] / df['Number of employees'].replace(0, np.nan)

    
    # Extract number of founders
    df['Number of Founders'] = df['Founders'].str.count(',') + 1
    
    # Clean and process investors
    df['Number of investors'] = df['Number of investors'].astype(str).str.extract(r'(\d+)').astype(float)

    
    # Calculate funding efficiency
    df['Funding Efficiency'] = df['Funding'] / (df['Funding rounds'].replace(0, 1))
    
    # Create industry categories
    df['Industry Category'] = df['Industries'].apply(lambda x: x.split(',')[0] if pd.notna(x) else 'Unknown')
    
    return df

# Load data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()
# Navigation
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>Startup Insights</h1>", unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=["Overview", "Market Analysis", "Funding Insights", "Founder Network", "Predictive Analytics", "Competitive Analysis", "Investment Opportunities"],
        icons=["house", "globe", "cash-coin", "people-fill", "graph-up", "bar-chart-fill", "gem"],
        menu_icon="cast",
        default_index=0,
    )
    
    st.markdown("---")
    
    # Filters
    st.subheader("Filter Startups")
    
    # Dynamic filters based on data
    country_filter = st.multiselect("Select Country", df['Country'].unique())
    industry_filter = st.multiselect("Select Industry", df['Industries'].unique())
    
    # Date range filter
    year_range = st.slider(
        "Founded Between",
        int(df['Started in'].min()),
        int(df['Started in'].max()),
        (int(df['Started in'].min()), int(df['Started in'].max()))
    )
    
    # Funding range filter
    funding_range = st.slider(
        "Funding Range ($)",
        float(df['Funding'].min()),
        float(df['Funding'].max()),
        (float(df['Funding'].min()), float(df['Funding'].max())),
        format="$%.0f"
    )
    
    # Apply filters
    filtered_df = df.copy()
    if country_filter:
        filtered_df = filtered_df[filtered_df['Country'].isin(country_filter)]
    if industry_filter:
        filtered_df = filtered_df[filtered_df['Industries'].isin(industry_filter)]
    filtered_df = filtered_df[(filtered_df['Started in'] >= year_range[0]) & (filtered_df['Started in'] <= year_range[1])]
    filtered_df = filtered_df[(filtered_df['Funding'] >= funding_range[0]) & (filtered_df['Funding'] <= funding_range[1])]
    
    # Reset filters button
    if st.button("Reset Filters"):
        country_filter = []
        industry_filter = []
        year_range = (int(df['Started in'].min()), int(df['Started in'].max()))
        funding_range = (float(df['Funding'].min()), float(df['Funding'].max()))
        filtered_df = df.copy()
    
    st.markdown("---")
    st.markdown("**Developed by Deepak and Team**")


# Content based on navigation
if selected == "Overview":
    st.markdown("<h1 class='main-header'>🚀 Startup Ecosystem Overview</h1>", unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{len(filtered_df):,}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Startups</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>${filtered_df['Funding'].sum()/1_000_000_000:.2f}B</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Funding</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{filtered_df['Country'].nunique()}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Countries</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>${filtered_df['Funding'].median()/1_000_000:.1f}M</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Median Funding</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>Quick Insights</h2>", unsafe_allow_html=True)
    
    # Top funded startups
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Top 10 Funded Startups</h3>", unsafe_allow_html=True)
        top_funded = filtered_df.nlargest(10, 'Funding')
        fig = px.bar(
            top_funded,
            x='Startup Name',
            y='Funding',
            color='Industries',
            hover_data=['Country', 'Started in'],
            title="",
            labels={'Funding': 'Funding ($)'},
            height=400
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Industry Distribution</h3>", unsafe_allow_html=True)
        industry_counts = filtered_df['Industry Category'].value_counts().nlargest(10)
        fig2 = px.pie(
            values=industry_counts,
            names=industry_counts.index,
            title="",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Geographic distribution
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Startup Geographic Distribution</h3>", unsafe_allow_html=True)
    geo_counts = filtered_df['Country'].value_counts().reset_index()
    geo_counts.columns = ['Country', 'Count']
    
    fig3 = px.choropleth(
        geo_counts,
        locations='Country',
        locationmode='country names',
        color='Count',
        hover_name='Country',
        color_continuous_scale=px.colors.sequential.Blues,
        title=""
    )
    fig3.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        )
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Timeline of startup founding
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Startup Founding Timeline</h3>", unsafe_allow_html=True)
    timeline_data = filtered_df.groupby('Started in').size().reset_index(name='Count')
    fig4 = px.line(
        timeline_data,
        x='Started in',
        y='Count',
        title="",
        labels={'Started in': 'Year', 'Count': 'Number of Startups Founded'},
        markers=True
    )
    fig4.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Market Analysis":
    st.markdown("<h1 class='main-header'>🌍 Market Analysis</h1>", unsafe_allow_html=True)
    
    # Market trends
    st.markdown("<h2 class='sub-header'>Industry Trends</h2>", unsafe_allow_html=True)
    
    # Industry growth over time
    industry_growth = filtered_df.groupby(['Started in', 'Industry Category']).size().reset_index(name='Count')
    industry_growth = industry_growth.pivot(index='Started in', columns='Industry Category', values='Count').fillna(0)
    
    # Get top 5 industries
    top_industries = filtered_df['Industry Category'].value_counts().nlargest(5).index.tolist()
    industry_growth_top = industry_growth[top_industries].reset_index()
    
    fig = px.line(
        industry_growth_top,
        x='Started in',
        y=top_industries,
        title="Growth of Top 5 Industries Over Time",
        labels={'Started in': 'Year', 'value': 'Number of Startups'},
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Industry funding comparison
    st.markdown("<h2 class='sub-header'>Industry Funding Comparison</h2>", unsafe_allow_html=True)
    
    industry_funding = filtered_df.groupby('Industry Category')['Funding'].agg(['sum', 'mean', 'median', 'count']).reset_index()
    industry_funding = industry_funding.sort_values('sum', ascending=False).head(10)
    
    fig2 = px.bar(
        industry_funding,
        x='Industry Category',
        y='sum',
        color='count',
        text='count',
        title="Top 10 Industries by Total Funding",
        labels={'sum': 'Total Funding ($)', 'count': 'Number of Startups', 'Industry Category': 'Industry'},
        height=500,
        color_continuous_scale=px.colors.sequential.Blues
    )
    fig2.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Regional market analysis
    st.markdown("<h2 class='sub-header'>Regional Market Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top countries by number of startups
        country_counts = filtered_df['Country'].value_counts().nlargest(10).reset_index()
        country_counts.columns = ['Country', 'Count']
        
        fig3 = px.bar(
            country_counts,
            x='Country',
            y='Count',
            title="Top 10 Countries by Number of Startups",
            color='Count',
            color_continuous_scale=px.colors.sequential.Blues
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Top countries by total funding
        country_funding = filtered_df.groupby('Country')['Funding'].sum().nlargest(10).reset_index()
        
        fig4 = px.bar(
            country_funding,
            x='Country',
            y='Funding',
            title="Top 10 Countries by Total Funding",
            color='Funding',
            color_continuous_scale=px.colors.sequential.Greens,
            labels={'Funding': 'Total Funding ($)'}
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    # Market segment analysis
    st.markdown("<h2 class='sub-header'>Market Segment Analysis</h2>", unsafe_allow_html=True)
    
    # Bubble chart of funding vs. number of startups by industry
    industry_analysis = filtered_df.groupby('Industry Category').agg({
        'Funding': 'sum',
        'Startup Name': 'count',
        'Age': 'mean'
    }).reset_index()
    
    industry_analysis.columns = ['Industry', 'Total Funding', 'Number of Startups', 'Average Age']
    
    fig5 = px.scatter(
        industry_analysis,
        x='Number of Startups',
        y='Total Funding',
        size='Average Age',
        color='Industry',
        hover_name='Industry',
        title="Industry Landscape: Funding vs. Popularity",
        labels={'Total Funding': 'Total Funding ($)'},
        height=600
    )
    st.plotly_chart(fig5, use_container_width=True)

elif selected == "Funding Insights":
    st.markdown("<h1 class='main-header'>💰 Funding Insights</h1>", unsafe_allow_html=True)
    
    # Funding metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>${filtered_df['Funding'].sum()/1_000_000_000:.2f}B</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Funding</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>${filtered_df['Funding'].mean()/1_000_000:.1f}M</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Average Funding</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        avg_rounds = filtered_df['Funding rounds'].mean()
        st.markdown(f"<div class='metric-value'>{avg_rounds:.1f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Avg Funding Rounds</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        efficiency = filtered_df['Funding Efficiency'].mean() / 1_000_000
        st.markdown(f"<div class='metric-value'>${efficiency:.1f}M</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Avg $ per Round</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Funding distribution
    st.markdown("<h2 class='sub-header'>Funding Distribution</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram of funding
        fig = px.histogram(
            filtered_df,
            x='Funding',
            nbins=30,
            title="Funding Distribution (Log Scale)",
            log_x=True,
            color_discrete_sequence=['#1E88E5']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot by industry
        top_industries = filtered_df['Industry Category'].value_counts().nlargest(5).index.tolist()
        industry_funding = filtered_df[filtered_df['Industry Category'].isin(top_industries)]
        
        fig2 = px.box(
            industry_funding,
            x='Industry Category',
            y='Funding',
            title="Funding Distribution by Top Industries",
            color='Industry Category',
            log_y=True,
            labels={'Funding': 'Funding ($)', 'Industry Category': 'Industry'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Funding trends over time
    st.markdown("<h2 class='sub-header'>Funding Trends Over Time</h2>", unsafe_allow_html=True)
    
    # Line chart of total funding by year
    funding_by_year = filtered_df.groupby('Started in')['Funding'].sum().reset_index()
    count_by_year = filtered_df.groupby('Started in').size().reset_index(name='Count')
    funding_trend = pd.merge(funding_by_year, count_by_year, on='Started in')
    funding_trend['Average Funding'] = funding_trend['Funding'] / funding_trend['Count']
    
    fig3 = px.line(
        funding_trend,
        x='Started in',
        y=['Funding', 'Average Funding'],
        title="Funding Trends Over Time",
        labels={'Started in': 'Year', 'value': 'Funding ($)', 'variable': 'Metric'},
        markers=True
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Investor analysis
    st.markdown("<h2 class='sub-header'>Investor Analysis</h2>", unsafe_allow_html=True)
    
    # Scatter plot of funding vs. number of investors
    fig4 = px.scatter(
        filtered_df,
        x='Number of investors',
        y='Funding',
        color='Industry Category',
        size='Funding rounds',
        hover_name='Startup Name',
        title="Funding vs. Number of Investors",
        labels={'Funding': 'Funding ($)', 'Number of investors': 'Number of Investors'},
        log_y=True
    )
    st.plotly_chart(fig4, use_container_width=True)
    
    # Funding efficiency
    st.markdown("<h2 class='sub-header'>Funding Efficiency</h2>", unsafe_allow_html=True)
    
    # Calculate funding efficiency metrics
    filtered_df['Funding per Employee'] = filtered_df['Funding'] / filtered_df['Number of employees'].replace(0, np.nan)
    filtered_df['Funding per Year'] = filtered_df['Funding'] / filtered_df['Age'].replace(0, 1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Funding per employee by industry
        industry_efficiency = filtered_df.groupby('Industry Category')['Funding per Employee'].median().nlargest(10).reset_index()
        
        fig5 = px.bar(
            industry_efficiency,
            x='Industry Category',
            y='Funding per Employee',
            title="Median Funding per Employee by Industry",
            color='Funding per Employee',
            labels={'Funding per Employee': 'Funding per Employee ($)', 'Industry Category': 'Industry'},
            color_continuous_scale=px.colors.sequential.Blues
        )
        fig5.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig5, use_container_width=True)
    
    with col2:
        # Funding per year by industry
        industry_time_efficiency = filtered_df.groupby('Industry Category')['Funding per Year'].median().nlargest(10).reset_index()
        
        fig6 = px.bar(
            industry_time_efficiency,
            x='Industry Category',
            y='Funding per Year',
            title="Median Funding per Year by Industry",
            color='Funding per Year',
            labels={'Funding per Year': 'Funding per Year ($)', 'Industry Category': 'Industry'},
            color_continuous_scale=px.colors.sequential.Greens
        )
        fig6.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig6, use_container_width=True)

elif selected == "Founder Network":
    st.markdown("<h1 class='main-header'>👥 Founder Network Analysis</h1>", unsafe_allow_html=True)
    
    # Founder metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        total_founders = filtered_df['Number of Founders'].sum()
        st.markdown(f"<div class='metric-value'>{int(total_founders):,}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Founders</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        avg_founders = filtered_df['Number of Founders'].mean()
        st.markdown(f"<div class='metric-value'>{avg_founders:.1f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Avg Founders per Startup</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        solo_founders = (filtered_df['Number of Founders'] == 1).sum()
        solo_percent = (solo_founders / len(filtered_df)) * 100
        st.markdown(f"<div class='metric-value'>{solo_percent:.1f}%</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Solo Founders</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Founder team size analysis
    st.markdown("<h2 class='sub-header'>Founder Team Size Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution of team sizes
        team_size_dist = filtered_df['Number of Founders'].value_counts().sort_index().reset_index()
        team_size_dist.columns = ['Team Size', 'Count']
        
        fig = px.bar(
            team_size_dist,
            x='Team Size',
            y='Count',
            title="Distribution of Founder Team Sizes",
            color='Count',
            labels={'Team Size': 'Number of Founders', 'Count': 'Number of Startups'},
            color_continuous_scale=px.colors.sequential.Blues
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Team size vs funding
        team_funding = filtered_df.groupby('Number of Founders')['Funding'].median().reset_index()
        
        fig2 = px.line(
            team_funding,
            x='Number of Founders',
            y='Funding',
            title="Median Funding by Team Size",
            markers=True,
            labels={'Funding': 'Median Funding ($)', 'Number of Founders': 'Team Size'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Founder network visualization
    st.markdown("<h2 class='sub-header'>Founder Network Visualization</h2>", unsafe_allow_html=True)
    
    # Sample of data for visualization
    sample_df = filtered_df.sample(min(50, len(filtered_df)))
    
    # Create network graph
    G = nx.Graph()
    
    for index, row in sample_df.iterrows():
        founders = str(row['Founders']).split(', ')
        startup = row['Startup Name']
        industry = row['Industry Category']
        
        # Add startup as a node
        G.add_node(startup, type='startup', industry=industry)
        
        # Add founders as nodes and connect to startup
        for founder in founders:
            if founder != 'nan':
                G.add_node(founder, type='founder')
                G.add_edge(founder, startup)
    
    # Layout
    pos = nx.spring_layout(G, k=0.3, iterations=50)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Draw startup nodes
    startup_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get('type') == 'startup']
    nx.draw_networkx_nodes(G, pos, nodelist=startup_nodes, node_size=100, node_color='skyblue', alpha=0.8)
    
    # Draw founder nodes
    founder_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get('type') == 'founder']
    nx.draw_networkx_nodes(G, pos, nodelist=founder_nodes, node_size=50, node_color='orange', alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
    
    plt.title("Founder and Startup Network")
    plt.axis('off')
    st.pyplot(plt)
    
    # Founder success metrics
    st.markdown("<h2 class='sub-header'>Founder Success Metrics</h2>", unsafe_allow_html=True)
    
    # Correlation between team size and funding
    col1, col2 = st.columns(2)
    
    with col1:
        # Team size vs funding efficiency
        team_efficiency = filtered_df.groupby('Number of Founders')['Funding Efficiency'].median().reset_index()
        
        fig3 = px.bar(
            team_efficiency,
            x='Number of Founders',
            y='Funding Efficiency',
            title="Funding Efficiency by Team Size",
            color='Funding Efficiency',
               labels={'Funding Efficiency': 'Funding per Round ($)', 'Number of Founders': 'Team Size'},
            color_continuous_scale=px.colors.sequential.Greens
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Team size vs company age
        team_age = filtered_df.groupby('Number of Founders')['Age'].mean().reset_index()
        
        fig4 = px.bar(
            team_age,
            x='Number of Founders',
            y='Age',
            title="Average Company Age by Team Size",
            color='Age',
            labels={'Age': 'Average Age (Years)', 'Number of Founders': 'Team Size'},
            color_continuous_scale=px.colors.sequential.Blues
        )
        st.plotly_chart(fig4, use_container_width=True)

elif selected == "Predictive Analytics":
    st.markdown("<h1 class='main-header'>📈 Predictive Analytics</h1>", unsafe_allow_html=True)
    
    # Data preparation for models
    st.markdown("<h2 class='sub-header'>Funding Prediction Model</h2>", unsafe_allow_html=True)
    
    # Prepare data for modeling
    model_df = filtered_df.copy()
    
    # Select features and target
    features = ['Started in', 'Number of Founders', 'Number of employees', 'Funding rounds', 'Number of investors']
    categorical_features = ['Country', 'Industry Category']
    

    
    # Encode categorical features
    le_dict = {}
    for feature in categorical_features:
        le = LabelEncoder()
        model_df[f'{feature}_encoded'] = le.fit_transform(model_df[feature].astype(str))
        le_dict[feature] = le
    
    # Combine features
    all_features = features + [f'{feature}_encoded' for feature in categorical_features]
    X = model_df[all_features]
    y = model_df['Funding']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Model performance
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>${mae/1_000_000:.2f}M</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Mean Absolute Error</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{r2:.2f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>R² Score</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': all_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(
        feature_importance,
        x='Feature',
        y='Importance',
        title="Feature Importance for Funding Prediction",
        color='Importance',
        labels={'Importance': 'Importance Score'},
        color_continuous_scale=px.colors.sequential.Blues
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive prediction
    st.markdown("<h2 class='sub-header'>Interactive Funding Prediction</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        industry_input = st.selectbox("Select Industry", model_df['Industry Category'].unique())
        country_input = st.selectbox("Select Country", model_df['Country'].unique())
        year_input = st.slider("Select Year Founded", int(model_df['Started in'].min()), int(model_df['Started in'].max()), 2020)
    
    with col2:
        team_size = st.slider("Number of Founders", 1, 10, 2)
        employees = st.slider("Number of Employees", 1, 1000, 50)
        funding_rounds = st.slider("Funding Rounds", 1, 10, 2)
        investors = st.slider("Number of Investors", 1, 20, 5)
    
    # Predict button
    if st.button("Predict Funding"):
        # Encode inputs
        industry_encoded = le_dict['Industry Category'].transform([industry_input])[0]
        country_encoded = le_dict['Country'].transform([country_input])[0]
        
        # Create input array
        input_data = np.array([[
            year_input, team_size, employees, funding_rounds, investors,
            country_encoded, industry_encoded
        ]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display prediction
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Funding Prediction</h3>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 2.5rem; font-weight: 700; color: #1E88E5; text-align: center;'>${prediction/1_000_000:.2f}M</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Success prediction model
    st.markdown("<h2 class='sub-header'>Startup Success Prediction</h2>", unsafe_allow_html=True)
    
    # Define success threshold
    success_threshold = model_df['Funding'].median()
    model_df['Success'] = model_df['Funding'].apply(lambda x: 1 if x > success_threshold else 0)
    model_df = model_df.dropna(subset=all_features + ['Success'])
    # Train success prediction model
    X_success = model_df[all_features]
    y_success = model_df['Success']
    
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_success, y_success, test_size=0.2, random_state=42)
    
    success_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    success_model.fit(X_train_s, y_train_s)
    
    # Success prediction
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Success Probability Prediction</h3>", unsafe_allow_html=True)
    
    if st.button("Predict Success Probability"):
        # Encode inputs
        industry_encoded = le_dict['Industry Category'].transform([industry_input])[0]
        country_encoded = le_dict['Country'].transform([country_input])[0]
        
        # Create input array
        input_data = np.array([[
            year_input, team_size, employees, funding_rounds, investors,
            country_encoded, industry_encoded
        ]])
        
        # Make prediction
        success_prob = success_model.predict(input_data)[0]
        
        # Display prediction
        st.markdown(f"<div style='font-size: 2.5rem; font-weight: 700; color: #1E88E5; text-align: center;'>{success_prob*100:.1f}%</div>", unsafe_allow_html=True)
        
        # Recommendation
        st.markdown("<h4>Recommendations</h4>", unsafe_allow_html=True)
        
        if success_prob > 0.7:
            st.success("This startup has high potential for success. Consider investing or partnering.")
        elif success_prob > 0.4:
            st.warning("This startup has moderate potential. Further due diligence recommended.")
        else:
            st.error("This startup may face challenges. Consider alternative investments.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Funding trend forecasting
    st.markdown("<h2 class='sub-header'>Funding Trend Forecasting</h2>", unsafe_allow_html=True)
    
    # Prepare time series data
    time_series = model_df.groupby('Started in')['Funding'].sum().reset_index()
    time_series.columns = ['Year', 'Funding']
    
    # Fit time series model
    X_time = sm.add_constant(time_series['Year'])
    model_time = sm.OLS(time_series['Funding'], X_time).fit()
    
    # Make forecast
    future_years = np.array(range(int(time_series['Year'].max()) + 1, int(time_series['Year'].max()) + 6))
    future_X = sm.add_constant(future_years)
    forecast = model_time.predict(future_X)
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'Year': future_years,
        'Funding': forecast
    })
    
    # Combine historical and forecast
    combined_df = pd.concat([time_series, forecast_df])
    combined_df['Type'] = ['Historical'] * len(time_series) + ['Forecast'] * len(forecast_df)
    
    # Plot forecast
    fig = px.line(
        combined_df,
        x='Year',
        y='Funding',
        color='Type',
        title="Funding Trend Forecast",
        labels={'Funding': 'Total Funding ($)', 'Year': 'Year'},
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

elif selected == "Competitive Analysis":
    st.markdown("<h1 class='main-header'>🏆 Competitive Analysis</h1>", unsafe_allow_html=True)
    
    # Industry competitive landscape
    st.markdown("<h2 class='sub-header'>Industry Competitive Landscape</h2>", unsafe_allow_html=True)
    
    # Select industry for analysis
    industry_select = st.selectbox(
        "Select Industry for Analysis",
        filtered_df['Industry Category'].value_counts().nlargest(10).index.tolist()
    )
    
    # Filter data for selected industry
    industry_data = filtered_df[filtered_df['Industry Category'] == industry_select]
    
    # Top companies in industry
    st.markdown("<h3>Top Companies in Industry</h3>", unsafe_allow_html=True)
    top_companies = industry_data.nlargest(10, 'Funding')
    
    fig = px.bar(
        top_companies,
        x='Startup Name',
        y='Funding',
        title=f"Top 10 Funded Companies in {industry_select}",
        color='Funding',
        labels={'Funding': 'Funding ($)', 'Startup Name': 'Company'},
        color_continuous_scale=px.colors.sequential.Blues
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Competitive metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{len(industry_data)}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Companies</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>${industry_data['Funding'].sum()/1_000_000_000:.2f}B</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Industry Funding</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{industry_data['Country'].nunique()}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Countries</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Market concentration
    st.markdown("<h3>Market Concentration</h3>", unsafe_allow_html=True)
    
    # Calculate market concentration
    total_funding = industry_data['Funding'].sum()
    industry_data['Market Share'] = industry_data['Funding'] / total_funding * 100
    
    # Top 5 concentration
    top5_concentration = industry_data.nlargest(5, 'Funding')['Market Share'].sum()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{top5_concentration:.1f}%</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Top 5 Market Share</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Herfindahl-Hirschman Index (HHI)
        hhi = (industry_data['Market Share'] ** 2).sum()
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{hhi:.0f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Market Concentration (HHI)</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Market share visualization
    market_share = industry_data.nlargest(10, 'Funding')
    
    fig2 = px.pie(
        market_share,
        values='Funding',
        names='Startup Name',
        title=f"Market Share Distribution in {industry_select}",
        hole=0.4
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Competitor clustering
    st.markdown("<h3>Competitor Clustering</h3>", unsafe_allow_html=True)
    
    # Prepare data for clustering
    cluster_data = industry_data.copy()
    
    # Select features for clustering
    cluster_features = ['Funding', 'Started in', 'Number of employees', 'Funding rounds', 'Number of investors']
    
    # Standardize features
    scaler = StandardScaler()
    cluster_data_scaled = scaler.fit_transform(cluster_data[cluster_features].fillna(0))
    
    # Perform clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_data['Cluster'] = kmeans.fit_predict(cluster_data_scaled)
    
    # Visualize clusters
    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    cluster_data_pca = pca.fit_transform(cluster_data_scaled)
    cluster_data['PCA1'] = cluster_data_pca[:, 0]
    cluster_data['PCA2'] = cluster_data_pca[:, 1]
    
    fig3 = px.scatter(
        cluster_data,
        x='PCA1',
        y='PCA2',
        color='Cluster',
        hover_name='Startup Name',
        hover_data=['Funding', 'Started in', 'Number of employees'],
        title=f"Competitor Clusters in {industry_select}",
        labels={'PCA1': 'Component 1', 'PCA2': 'Component 2'}
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Cluster characteristics
    st.markdown("<h3>Cluster Characteristics</h3>", unsafe_allow_html=True)
    
    cluster_stats = cluster_data.groupby('Cluster').agg({
        'Funding': 'mean',
        'Started in': 'mean',
        'Number of employees': 'mean',
        'Funding rounds': 'mean',
        'Number of investors': 'mean',
        'Startup Name': 'count'
    }).reset_index()
    
    cluster_stats.columns = ['Cluster', 'Avg Funding', 'Avg Year Founded', 'Avg Employees', 'Avg Funding Rounds', 'Avg Investors', 'Number of Companies']
    
    st.dataframe(cluster_stats)
    
    # Competitive positioning
    st.markdown("<h3>Competitive Positioning</h3>", unsafe_allow_html=True)
    
    # Select dimensions for positioning
    x_axis = st.selectbox("X-Axis", ['Funding', 'Number of employees', 'Started in', 'Funding rounds', 'Number of investors'])
    y_axis = st.selectbox("Y-Axis", ['Number of employees', 'Funding', 'Started in', 'Funding rounds', 'Number of investors'])
    
    fig4 = px.scatter(
        industry_data,
        x=x_axis,
        y=y_axis,
        color='Country',
        size='Funding',
        hover_name='Startup Name',
        title=f"Competitive Positioning in {industry_select}",
        labels={x_axis: x_axis, y_axis: y_axis}
    )
    st.plotly_chart(fig4, use_container_width=True)

elif selected == "Investment Opportunities":
    st.markdown("<h1 class='main-header'>💎 Investment Opportunities</h1>", unsafe_allow_html=True)
    
    # Investment opportunity finder
    st.markdown("<h2 class='sub-header'>Investment Opportunity Finder</h2>", unsafe_allow_html=True)
    
    # Investment criteria
    st.markdown("<h3>Set Investment Criteria</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_funding = st.slider("Minimum Funding ($)", 0, 10000000, 1000000, 100000)
        max_funding = st.slider("Maximum Funding ($)", min_funding, 100000000, 10000000, 1000000)
        min_year = st.slider("Minimum Year Founded", int(filtered_df['Started in'].min()), int(filtered_df['Started in'].max()), 2015)
    
    with col2:
        selected_industries = st.multiselect("Target Industries", filtered_df['Industry Category'].unique())
        selected_countries = st.multiselect("Target Countries", filtered_df['Country'].unique())
        min_employees = st.slider("Minimum Employees", 1, 1000, 10)
    
    # Apply filters
    opportunity_df = filtered_df.copy()
    
    if min_funding > 0:
        opportunity_df = opportunity_df[opportunity_df['Funding'] >= min_funding]
    if max_funding < 100000000:
        opportunity_df = opportunity_df[opportunity_df['Funding'] <= max_funding]
    if min_year > int(filtered_df['Started in'].min()):
        opportunity_df = opportunity_df[opportunity_df['Started in'] >= min_year]
    if selected_industries:
        opportunity_df = opportunity_df[opportunity_df['Industry Category'].isin(selected_industries)]
    if selected_countries:
        opportunity_df = opportunity_df[opportunity_df['Country'].isin(selected_countries)]
    if min_employees > 1:
        opportunity_df = opportunity_df[opportunity_df['Number of employees'] >= min_employees]
    
    # Check if opportunity_df is empty after filtering
    if opportunity_df.shape[0] > 0:
        # Calculate investment score
        opportunity_df['Age'] = datetime.now().year - opportunity_df['Started in']
        opportunity_df['Funding per Employee'] = opportunity_df['Funding'] / opportunity_df['Number of employees'].replace(0, 1)
        opportunity_df['Funding per Year'] = opportunity_df['Funding'] / opportunity_df['Age'].replace(0, 1)
        opportunity_df['Funding Efficiency'] = opportunity_df['Funding'] / opportunity_df['Funding rounds'].replace(0, 1)
        
        # Normalize metrics
        for metric in ['Funding per Employee', 'Funding per Year', 'Funding Efficiency']:
            min_val = opportunity_df[metric].min()
            max_val = opportunity_df[metric].max()
            opportunity_df[f'{metric}_norm'] = (opportunity_df[metric] - min_val) / (max_val - min_val)
        
        # Calculate investment score
        opportunity_df['Investment Score'] = (
            opportunity_df['Funding per Employee_norm'] * 0.3 +
            opportunity_df['Funding per Year_norm'] * 0.3 +
            opportunity_df['Funding Efficiency_norm'] * 0.4
        )
        
        # Display top opportunities
        st.markdown("<h3>Top Investment Opportunities</h3>", unsafe_allow_html=True)
        
        top_opportunities = opportunity_df.nlargest(10, 'Investment Score')
        
        st.dataframe(
            top_opportunities[['Startup Name', 'Industry Category', 'Country', 'Funding', 'Started in', 'Number of employees', 'Investment Score']]
        )
        
        # Investment opportunity details
        st.markdown("<h3>Opportunity Details</h3>", unsafe_allow_html=True)
        
        # Select startup for detailed view
        selected_startup = st.selectbox("Select Startup for Details", top_opportunities['Startup Name'])
        
        # Ensure the selected startup exists in the filtered data
        if selected_startup in top_opportunities['Startup Name'].values:
            # Display details
            startup_details = opportunity_df[opportunity_df['Startup Name'] == selected_startup].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"<h3>{startup_details['Startup Name']}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Industry:</strong> {startup_details['Industry Category']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Country:</strong> {startup_details['Country']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Founded:</strong> {int(startup_details['Started in'])}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Funding:</strong> ${startup_details['Funding']:,.2f}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Employees:</strong> {startup_details['Number of employees']}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h3>Investment Metrics</h3>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Investment Score:</strong> {startup_details['Investment Score']:.2f}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Funding per Employee:</strong> ${startup_details['Funding per Employee']:,.2f}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Funding per Year:</strong> ${startup_details['Funding per Year']:,.2f}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Funding Efficiency:</strong> ${startup_details['Funding Efficiency']:,.2f}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Funding Rounds:</strong> {startup_details['Funding rounds']}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("The selected startup is not available in the filtered list. Please adjust your filters.")
    else:
        st.error("No investment opportunities found with the current filters.")
    
    # Investment portfolio simulator
    st.markdown("<h2 class='sub-header'>Investment Portfolio Simulator</h2>", unsafe_allow_html=True)
    
    # Portfolio settings
    col1, col2 = st.columns(2)
    
    with col1:
        portfolio_size = st.slider("Portfolio Size (Number of Startups)", 1, 10, 5)
        investment_amount = st.number_input("Total Investment Amount ($)", 1000000, 100000000, 10000000, 1000000)
    
    with col2:
        risk_profile = st.select_slider("Risk Profile", options=["Conservative", "Balanced", "Aggressive"])
        diversification = st.checkbox("Geographic Diversification", True)
    
    # Generate portfolio
    if st.button("Generate Portfolio"):
        st.markdown("<h3>Simulated Investment Portfolio</h3>", unsafe_allow_html=True)
        
        # Filter based on risk profile
        if risk_profile == "Conservative":
            risk_filtered = opportunity_df[opportunity_df['Started in'] <= 2018]
        elif risk_profile == "Balanced":
            risk_filtered = opportunity_df[opportunity_df['Started in'] > 2015]
        else:  # Aggressive
            risk_filtered = opportunity_df[opportunity_df['Started in'] > 2018]
        
        # Apply diversification if selected
        if diversification:
            # Get top countries
            top_countries = risk_filtered['Country'].value_counts().nlargest(portfolio_size).index.tolist()
            portfolio_companies = []
            
            # Select top company from each country
            for country in top_countries[:portfolio_size]:
                country_companies = risk_filtered[risk_filtered['Country'] == country]
                if not country_companies.empty:
                    top_company = country_companies.nlargest(1, 'Investment Score')
                    portfolio_companies.append(top_company)
            if portfolio_companies:
                portfolio = pd.concat(portfolio_companies)
            else:
                st.error("No companies found for the selected diversification criteria.")
                portfolio = pd.DataFrame() 
        else:
            portfolio = risk_filtered.nlargest(portfolio_size, 'Investment Score')
        
        if portfolio.empty:
            st.error("No companies found matching the portfolio criteria. Please adjust your filters.")
        else:
    # Calculate investment allocation
            portfolio['Allocation'] = portfolio['Investment Score'] / portfolio['Investment Score'].sum()
            portfolio['Investment'] = portfolio['Allocation'] * investment_amount
    
        
        # Display portfolio
        st.dataframe(
            portfolio[['Startup Name', 'Industry Category', 'Country', 'Investment Score', 'Allocation', 'Investment']]
        )
        
        # Portfolio visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Industry allocation
            industry_allocation = portfolio.groupby('Industry Category')['Investment'].sum().reset_index()
            
            fig = px.pie(
                industry_allocation,
                values='Investment',
                names='Industry Category',
                title="Industry Allocation",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Country allocation
            country_allocation = portfolio.groupby('Country')['Investment'].sum().reset_index()
            
            fig2 = px.pie(
                country_allocation,
                values='Investment',
                names='Country',
                title="Country Allocation",
                hole=0.4
            )
            st.plotly_chart(fig2, use_container_width=True)
