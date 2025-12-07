"""
Hamburg Circular Economy Ecosystem Visualization
Interactive Streamlit application for exploring the Hamburg CE ecosystem
Part of the AI-InnoScEnCE Project
"""

import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import ast
import sqlite3
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Hamburg CE Ecosystem | AI-InnoScEnCE",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - AI-InnoScEnCE Branding (Light Theme)
st.markdown("""
<style>
    /* Main content area */
    .main {
        background-color: #FFFFFF;
    }

    /* Headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666666;
        margin-bottom: 1rem;
    }

    /* Metric cards */
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00BCD4;
    }
    .stMetric {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }

    /* Buttons - Turquoise theme */
    .stButton>button {
        background-color: #00BCD4;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00ACC1;
        border: none;
        box-shadow: 0 4px 8px rgba(0,188,212,0.3);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F5F5F5;
        padding: 0.5rem;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #1e3a5f;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00BCD4;
        color: white !important;
        border-radius: 6px;
    }

    /* Divider */
    .divider {
        margin: 2rem 0;
        border-top: 2px solid #e0e0e0;
    }

    /* Insight cards */
    .insight-card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #00BCD4;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        word-wrap: break-word;
        overflow-wrap: break-word;
        white-space: normal;
        max-height: none;
        overflow: visible;
    }
    .gap-card {
        border-left-color: #ff6b6b;
        background-color: #fff5f5;
    }
    .synergy-card {
        border-left-color: #4ecdc4;
        background-color: #f0fffe;
    }
    .recommendation-card {
        border-left-color: #95e1d3;
        background-color: #f7fffe;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #F5F5F5;
    }

    /* Dataframes and tables */
    .stDataFrame {
        background-color: #FFFFFF;
    }

    /* Headers in markdown */
    h1, h2, h3 {
        color: #1e3a5f;
    }
</style>
""", unsafe_allow_html=True)

# Database connectivity
@st.cache_resource
def get_db_connection():
    """Get SQLite database connection"""
    app_dir = Path(__file__).parent
    db_path = app_dir / ".." / "CE-Ecosystem Builder" / "hamburg_ce_ecosystem" / "data" / "final" / "ecosystem.db"
    db_path = db_path.resolve()

    if not db_path.exists():
        st.error(f"Database not found at: {db_path}")
        return None

    return sqlite3.connect(str(db_path), check_same_thread=False)

@st.cache_data
def load_relationships():
    """Load relationships from database"""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()

    query = """
    SELECT source_entity, target_entity, relationship_type, confidence,
           evidence, bidirectional, source_url, target_url
    FROM relationships
    ORDER BY confidence DESC
    """
    df = pd.read_sql_query(query, conn)
    return df

@st.cache_data
def load_clusters():
    """Load clusters from database"""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()

    query = """
    SELECT cluster_id, cluster_name, cluster_type, description,
           entities, items, confidence
    FROM clusters
    ORDER BY cluster_type, confidence DESC
    """
    df = pd.read_sql_query(query, conn)
    return df

@st.cache_data
def load_insights():
    """Load ecosystem insights from database"""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()

    query = """
    SELECT insight_type, title, description, entities_involved,
           confidence, priority, timestamp
    FROM ecosystem_insights
    ORDER BY
        CASE priority
            WHEN 'high' THEN 1
            WHEN 'medium' THEN 2
            ELSE 3
        END,
        confidence DESC
    """
    df = pd.read_sql_query(query, conn)
    return df

# Load data
@st.cache_data
def load_data(csv_path):
    """Load and preprocess the ecosystem entity data"""
    df = pd.read_csv(csv_path)

    # Clean up data
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    # Parse list-like strings
    def safe_parse_list(val):
        if pd.isna(val) or val in ['[]', '', 'NA']:
            return []
        try:
            return ast.literal_eval(val) if isinstance(val, str) else val
        except:
            return []

    for col in ['ce_activities', 'partners']:
        if col in df.columns:
            df[f'{col}_parsed'] = df[col].apply(safe_parse_list)
            df[f'{col}_count'] = df[f'{col}_parsed'].apply(len)

    # Clean entity names and roles
    df['entity_name'] = df['entity_name'].fillna('Unknown')
    df['ecosystem_role'] = df['ecosystem_role'].fillna('Unknown')

    # EXCLUDE SENSITIVE FIELDS - Don't load or display them
    sensitive_fields = ['contact_persons', 'emails', 'phone_numbers']
    for field in sensitive_fields:
        if field in df.columns:
            df = df.drop(columns=[field])

    # Filter out invalid roles (data artifacts)
    valid_roles = [
        'Industry Partners', 'Higher Education Institutions', 'Startups and Entrepreneurs',
        'Non-Governmental Organizations', 'Research Institutes', 'Researchers',
        'Media and Communication Partners', 'Citizen Associations', 'End-Users',
        'Public Authorities', 'Knowledge and Innovation Communities', 'Students',
        'Policy Makers', 'Funding Bodies'
    ]
    df = df[df['ecosystem_role'].isin(valid_roles + ['Unknown', ''])]

    return df

# Define color mapping for ecosystem roles
ROLE_COLORS = {
    'Industry Partners': [0, 128, 255],  # Blue
    'Higher Education Institutions': [255, 140, 0],  # Dark Orange
    'Startups and Entrepreneurs': [233, 30, 99],  # Pink/Magenta
    'Non-Governmental Organizations': [155, 89, 182],  # Purple
    'Research Institutes': [231, 76, 60],  # Red
    'Researchers': [241, 196, 15],  # Yellow
    'Media and Communication Partners': [26, 188, 156],  # Turquoise
    'Citizen Associations': [230, 126, 34],  # Orange
    'End-Users': [149, 165, 166],  # Gray
    'Public Authorities': [52, 73, 94],  # Dark Blue
    'Knowledge and Innovation Communities': [142, 68, 173],  # Purple
    'Students': [255, 193, 7],  # Amber/Gold
    'Policy Makers': [192, 57, 43],  # Dark Red
    'Funding Bodies': [103, 58, 183],  # Deep Purple
    'Unknown': [200, 200, 200],  # Light Gray
}

def main():
    # Header with logo
    app_dir = Path(__file__).parent
    logo_path = app_dir / "AI-INNOCENSE-LOGO.png"

    col1, col2 = st.columns([1, 6])
    with col1:
        if logo_path.exists():
            st.image(str(logo_path), width=180)
    with col2:
        st.markdown('<p class="main-header">Hamburg Circular Economy Ecosystem</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI-Empowered Innovation in Natural Science and Engineering for the Circular Economy</p>', unsafe_allow_html=True)

    st.markdown("---")

    # Load data
    # Use path relative to this file's location
    data_path = app_dir / ".." / "CE-Ecosystem Builder" / "hamburg_ce_ecosystem" / "data" / "final" / "ecosystem_entities.csv"
    data_path = data_path.resolve()  # Convert to absolute path

    if not data_path.exists():
        st.error(f"Data file not found at: {data_path}")
        st.info("Please ensure the data pipeline has been run or adjust the path in app.py")
        return

    with st.spinner("Loading ecosystem data..."):
        df = load_data(data_path)
        relationships_df = load_relationships()
        clusters_df = load_clusters()
        insights_df = load_insights()

    # Create entity-to-URL mapping for relationships links
    entity_url_map = df.set_index('entity_name')['url'].to_dict() if 'url' in df.columns else {}

    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Role filter
    all_roles = sorted(df['ecosystem_role'].dropna().unique())
    selected_roles = st.sidebar.multiselect(
        "Ecosystem Roles",
        options=all_roles,
        default=all_roles  # Show ALL roles by default
    )

    # Apply filters
    filtered_df = df.copy()
    if selected_roles:
        filtered_df = filtered_df[filtered_df['ecosystem_role'].isin(selected_roles)]
    
    # Statistics
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Statistics")
    st.sidebar.metric("Total Entities", f"{len(df):,}")
    st.sidebar.metric("Filtered Entities", f"{len(filtered_df):,}")
    st.sidebar.metric("Relationships & Synergies", f"{len(relationships_df):,}")
    st.sidebar.metric("Clusters", f"{len(clusters_df):,}")
    st.sidebar.metric("Insights", f"{len(insights_df):,}")

    # Main content
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🗺️ Map View",
        "🔗 Relationships & Synergies",
        "💡 Insights",
        "🤝 Collaboration",
        "📊 Analytics",
        "📋 Data Table",
        "ℹ️ About"
    ])

    with tab1:
        show_map_view(filtered_df, clusters_df)

    with tab2:
        show_relationships_tab(df, relationships_df, entity_url_map)

    with tab3:
        show_insights_dashboard(insights_df)

    with tab4:
        show_collaboration_finder(df, relationships_df, clusters_df)

    with tab5:
        show_analytics(df, filtered_df, relationships_df, clusters_df)

    with tab6:
        show_data_table(filtered_df, clusters_df)

    with tab7:
        show_about(df, relationships_df, clusters_df, insights_df)

def show_map_view(df, clusters_df):
    """Display interactive map with cluster overlay"""
    st.header("Interactive Hamburg CE Ecosystem Map")
    
    # Filter out rows without coordinates
    map_df = df.dropna(subset=['latitude', 'longitude'])
    
    if len(map_df) == 0:
        st.warning("No entities with coordinates in the current filter selection.")
        return
    
    # Add color based on role
    map_df['color'] = map_df['ecosystem_role'].map(lambda r: ROLE_COLORS.get(r, [200, 200, 200]))
    
    # Create tooltip with full CE activities list
    def format_tooltip(row):
        activities_list = row.get('ce_activities_parsed', [])
        if activities_list and len(activities_list) > 0:
            activities_html = '<br/>  • ' + '<br/>  • '.join(activities_list)
            activities_section = f"CE Activities ({len(activities_list)}):{activities_html}"
        else:
            activities_section = "CE Activities: None"

        return f"""
        <b>{row['entity_name']}</b><br/>
        Role: {row['ecosystem_role']}<br/>
        Location: Lat {row['latitude']:.6f}, Lon {row['longitude']:.6f}<br/>
        {activities_section}<br/>
        Partners: {row.get('partners_count', 0)}
        """

    map_df['tooltip'] = map_df.apply(format_tooltip, axis=1)
    
    # Map configuration
    view_state = pdk.ViewState(
        latitude=53.5511,
        longitude=9.9937,
        zoom=11,
        pitch=0,  # 0 = flat 2D view, 45 = 3D tilted view
    )
    
    # Layer configuration
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["longitude", "latitude"],
        get_color="color",
        get_radius=150,
        pickable=True,
        opacity=0.7,
        stroked=True,
        filled=True,
        radius_scale=1,
        radius_min_pixels=3,
        radius_max_pixels=50,
        line_width_min_pixels=1,
    )
    
    # Render map
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"html": "<b>{tooltip}</b>", "style": {"backgroundColor": "steelblue", "color": "white"}},
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",  # Free Carto basemap (no token required)
    )
    
    st.pydeck_chart(r, use_container_width=True)
    
    # Legend
    st.markdown("### Legend")
    cols = st.columns(5)
    for idx, (role, color) in enumerate(ROLE_COLORS.items()):
        if role in df['ecosystem_role'].values:
            count = len(df[df['ecosystem_role'] == role])
            cols[idx % 5].markdown(
                f'<div style="display: flex; align-items: center; margin: 0.2rem 0;">'
                f'<div style="width: 15px; height: 15px; min-width: 15px; min-height: 15px; '
                f'background-color: rgb({color[0]}, {color[1]}, {color[2]}); '
                f'border-radius: 50%; margin-right: 0.5rem; flex-shrink: 0; display: inline-block;"></div>'
                f'<span>{role} ({count})</span></div>',
                unsafe_allow_html=True
            )

def show_analytics(df, filtered_df, relationships_df, clusters_df):
    """Display analytics dashboard with relationships and cluster metrics"""
    st.header("Ecosystem Analytics")

    # Key metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Entities",
            f"{len(df):,}",
            delta=f"{len(filtered_df):,} filtered"
        )

    with col2:
        avg_ce_activities = df['ce_activities_count'].mean()
        st.metric(
            "Avg CE Activities",
            f"{avg_ce_activities:.1f}",
            delta=f"{df['ce_activities_count'].sum():,} total"
        )

    with col3:
        partnerships = len(relationships_df[relationships_df['relationship_type'] == 'partnership'])
        st.metric(
            "Partnerships",
            f"{partnerships:,}",
            delta=f"{len(relationships_df):,} total relationships"
        )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution by Ecosystem Role")
        role_counts = filtered_df['ecosystem_role'].value_counts().head(10)
        fig = px.bar(
            x=role_counts.values,
            y=role_counts.index,
            orientation='h',
            labels={'x': 'Count', 'y': 'Ecosystem Role'},
            color=role_counts.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("CE Activities Distribution")
        ce_bins = [0, 1, 3, 5, 10, 100]
        ce_labels = ['0', '1-2', '3-4', '5-9', '10+']
        df['ce_bin'] = pd.cut(df['ce_activities_count'], bins=ce_bins, labels=ce_labels, include_lowest=True)
        ce_dist = df['ce_bin'].value_counts().sort_index()
        fig = px.pie(
            values=ce_dist.values,
            names=ce_dist.index,
            title="Number of CE Activities per Entity"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Top entities by CE activities
    st.markdown("---")
    st.subheader("Top 10 Entities by CE Activities")
    top_ce = df.nlargest(10, 'ce_activities_count')[['entity_name', 'ecosystem_role', 'ce_activities_count', 'latitude', 'longitude']]
    top_ce = top_ce[top_ce['entity_name'] != 'Unknown']
    st.dataframe(top_ce, use_container_width=True, hide_index=True)

def show_data_table(df, clusters_df):
    """Display searchable data table"""
    st.header("Entity Data Explorer")

    # Search
    search = st.text_input("🔍 Search entities", placeholder="Enter entity name, role, or activity...")

    if search:
        mask = (
            df['entity_name'].str.contains(search, case=False, na=False) |
            df['ecosystem_role'].str.contains(search, case=False, na=False) |
            df['brief_description'].str.contains(search, case=False, na=False) |
            df['ce_activities'].str.contains(search, case=False, na=False)
        )
        df = df[mask]

    # Display options
    col1, col2 = st.columns(2)
    with col1:
        show_unknown = st.checkbox("Show 'Unknown' entities", value=False)
    with col2:
        # Exclude sensitive fields from column selection
        available_columns = ['entity_name', 'ecosystem_role', 'latitude', 'longitude',
                           'brief_description', 'ce_activities', 'partners', 'ce_relation']
        show_columns = st.multiselect(
            "Select columns to display",
            options=available_columns,
            default=['entity_name', 'ecosystem_role', 'latitude', 'longitude', 'ce_activities']
        )
    
    if not show_unknown:
        df = df[df['entity_name'] != 'Unknown']
    
    # Display table
    st.dataframe(
        df[show_columns].head(100),
        use_container_width=True,
        hide_index=True
    )
    
    st.info(f"Showing {min(100, len(df))} of {len(df)} entities. Use filters to narrow down results.")
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download filtered data as CSV",
        data=csv,
        file_name="hamburg_ce_ecosystem_filtered.csv",
        mime="text/csv"
    )

def show_about(df, relationships_df, clusters_df, insights_df):
    """Display about information"""
    st.header("About This Application")

    st.markdown("""
    ### Hamburg Circular Economy Ecosystem Visualizer

    This interactive application visualizes the comprehensive Hamburg Circular Economy (CE) ecosystem,
    built using **ScrapegraphAI** and advanced LLM-based extraction techniques as part of the
    **AI-InnoScEnCE Project**.

    #### 📊 Data Overview""")

    # Dynamic statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Entities", f"{len(df):,}")
        st.metric("Relationships & Synergies", f"{len(relationships_df):,}")
    with col2:
        partnerships = len(relationships_df[relationships_df['relationship_type'] == 'partnership'])
        st.metric("Partnerships", f"{partnerships:,}")
        st.metric("Clusters", f"{len(clusters_df):,}")
    with col3:
        st.metric("Insights", f"{len(insights_df):,}")

    st.markdown("""
    - Data extracted from: websites, Impressum pages, contact forms, and more
    - Privacy-compliant: No sensitive contact information is displayed
    
    #### 🎯 Ecosystem Roles

    The ecosystem comprises various stakeholders:
    - **Industry Partners** - Companies and manufacturers
    - **Higher Education Institutions** - Universities and colleges
    - **Startups and Entrepreneurs** - Innovative ventures and new businesses
    - **Non-Governmental Organizations** - NGOs and civil society organizations
    - **Research Institutes** - Research centers and laboratories
    - **Researchers** - Individual researchers and scientists
    - **Media and Communication Partners** - Press, media, and communication organizations
    - **Citizen Associations** - Community groups and local associations
    - **End-Users** - Consumers and end-users of circular products
    - **Public Authorities** - Government bodies and public agencies
    - **Knowledge and Innovation Communities** - KICs and innovation hubs
    - **Students** - Students and academic learners
    - **Policy Makers** - Policy developers and decision makers
    - **Funding Bodies** - Investors, grants, and funding organizations
    
    #### 🛠️ Technology Stack
    
    - **ScrapegraphAI** - AI-powered web scraping
    - **LLM Extraction** - Ollama-based structured extraction
    - **Geocoding** - Nominatim with intelligent fallback strategies
    - **Visualization** - Streamlit + PyDeck + Plotly

    #### 🔄 Data Pipeline

    1. **Verification** - LLM checks Hamburg & CE relevance
    2. **Extraction** - ScrapegraphAI + Ollama extract entity data
    3. **Geocoding** - Multi-strategy geocoding with caching
    4. **Visualization** - This Streamlit application

    #### 📧 Contact & Support

    For questions, suggestions, or collaboration opportunities, please visit:
    **[https://ai-innoscence.eu/](https://ai-innoscence.eu/)**

    ---

    **Built with ❤️ for a more circular Hamburg**
    """)

def show_relationships_tab(df, relationships_df, entity_url_map):
    """Display simple relationships view with partnerships and synergies"""
    st.header("🔗 Relationships & Synergies")

    st.markdown("""
    View partnerships and potential synergies within the Hamburg CE ecosystem.
    """)

    # Tabs for different relationship types
    rel_tabs = st.tabs(["🤝 Partnerships", "💡 Potential Synergies"])

    with rel_tabs[0]:
        show_partnerships_view(relationships_df, entity_url_map)

    with rel_tabs[1]:
        show_synergies_view(relationships_df, entity_url_map)


def show_partnerships_view(relationships_df, entity_url_map):
    """Display partnerships table"""
    st.subheader("Verified Partnerships")

    # Filter for partnerships only and exclude self-referencing relationships
    partnerships = relationships_df[relationships_df['relationship_type'] == 'partnership'].copy()
    partnerships = partnerships[partnerships['source_entity'] != partnerships['target_entity']]

    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Partnerships", len(partnerships))
    with col2:
        avg_conf = partnerships['confidence'].mean() if len(partnerships) > 0 else 0
        st.metric("Average Confidence", f"{avg_conf:.0%}")
    with col3:
        unique_entities = len(set(partnerships['source_entity'].tolist() + partnerships['target_entity'].tolist()))
        st.metric("Entities Involved", unique_entities)

    st.markdown("---")

    if len(partnerships) == 0:
        st.info("No partnerships found in the database.")
        return

    # Search functionality
    search = st.text_input("🔍 Search partnerships", placeholder="Search by entity name...")

    if search:
        mask = (
            partnerships['source_entity'].str.contains(search, case=False, na=False) |
            partnerships['target_entity'].str.contains(search, case=False, na=False) |
            partnerships['evidence'].str.contains(search, case=False, na=False)
        )
        partnerships = partnerships[mask]

    # Sort by confidence
    partnerships = partnerships.sort_values('confidence', ascending=False)

    # Display partnerships as expandable cards
    st.markdown(f"### {len(partnerships)} Partnership{'s' if len(partnerships) != 1 else ''}")

    for idx, row in partnerships.iterrows():
        with st.expander(f"**{row['source_entity']}** ↔ **{row['target_entity']}** | Confidence: {row['confidence']:.0%}"):
            st.markdown(f"**Evidence:**")
            st.write(row['evidence'])  # Full evidence text, not truncated

            # Use URLs from relationships table, fallback to entity URL mapping
            source_url = row.get('source_url') if pd.notna(row.get('source_url')) else entity_url_map.get(row['source_entity'])
            target_url = row.get('target_url') if pd.notna(row.get('target_url')) else entity_url_map.get(row['target_entity'])

            if source_url:
                st.markdown(f"🔗 [Visit {row['source_entity']}]({source_url})")
            if target_url:
                st.markdown(f"🔗 [Visit {row['target_entity']}]({target_url})")


def show_synergies_view(relationships_df, entity_url_map):
    """Display potential synergies table"""
    st.subheader("AI-Identified Potential Synergies")

    # Filter for synergies only and exclude self-referencing relationships
    synergies = relationships_df[relationships_df['relationship_type'] == 'potential_synergy'].copy()
    synergies = synergies[synergies['source_entity'] != synergies['target_entity']]

    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Synergies", f"{len(synergies):,}")
    with col2:
        avg_conf = synergies['confidence'].mean() if len(synergies) > 0 else 0
        st.metric("Average Confidence", f"{avg_conf:.0%}")
    with col3:
        high_conf = len(synergies[synergies['confidence'] >= 0.8])
        st.metric("High Confidence (≥80%)", f"{high_conf:,}")

    st.markdown("---")

    if len(synergies) == 0:
        st.info("No synergies found in the database.")
        return

    # Filters
    col1, col2 = st.columns(2)

    with col1:
        search = st.text_input("🔍 Search synergies", placeholder="Search by entity name...")

    with col2:
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.05,
            help="Filter synergies by minimum confidence score"
        )

    # Apply filters
    synergies = synergies[synergies['confidence'] >= min_confidence]

    if search:
        mask = (
            synergies['source_entity'].str.contains(search, case=False, na=False) |
            synergies['target_entity'].str.contains(search, case=False, na=False) |
            synergies['evidence'].str.contains(search, case=False, na=False)
        )
        synergies = synergies[mask]

    # Sort by confidence
    synergies = synergies.sort_values('confidence', ascending=False)

    # Limit display
    display_limit = st.select_slider(
        "Number of synergies to display",
        options=[10, 25, 50, 100, 250, 500],
        value=50
    )

    synergies_display = synergies.head(display_limit)

    st.markdown(f"### Showing {len(synergies_display):,} of {len(synergies):,} Synergies")

    # Display as expandable cards
    for idx, row in synergies_display.iterrows():
        # Color code confidence
        if row['confidence'] >= 0.9:
            conf_badge = "🟢 Very High"
        elif row['confidence'] >= 0.8:
            conf_badge = "🟡 High"
        else:
            conf_badge = "🔵 Medium"

        with st.expander(f"**{row['source_entity']}** ↔ **{row['target_entity']}** | {conf_badge} ({row['confidence']:.0%})"):
            st.markdown(f"**Evidence:**")
            st.write(row['evidence'])  # Full evidence text, not truncated

            # Use URLs from relationships table, fallback to entity URL mapping
            source_url = row.get('source_url') if pd.notna(row.get('source_url')) else entity_url_map.get(row['source_entity'])
            target_url = row.get('target_url') if pd.notna(row.get('target_url')) else entity_url_map.get(row['target_entity'])

            if source_url:
                st.markdown(f"🔗 [Visit {row['source_entity']}]({source_url})")
            if target_url:
                st.markdown(f"🔗 [Visit {row['target_entity']}]({target_url})")

def show_insights_dashboard(insights_df):
    """Display ecosystem insights dashboard"""
    st.header("💡 Ecosystem Insights")

    st.markdown("""
    AI-generated insights about the Hamburg CE ecosystem, including identified gaps,
    potential synergies, and recommendations for ecosystem development.
    """)

    if len(insights_df) == 0:
        st.warning("No insights available in the database.")
        return

    # Tabs for different insight types
    insight_tabs = st.tabs(["All Insights", "🔴 Gaps", "🤝 Synergies", "📋 Recommendations"])

    with insight_tabs[0]:
        show_all_insights(insights_df)

    with insight_tabs[1]:
        show_insights_by_type(insights_df, 'gap', 'gap-card')

    with insight_tabs[2]:
        show_insights_by_type(insights_df, 'synergy', 'synergy-card')

    with insight_tabs[3]:
        show_insights_by_type(insights_df, 'recommendation', 'recommendation-card')

def show_all_insights(insights_df):
    """Display all insights"""
    st.subheader("All Ecosystem Insights")

    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        gaps = len(insights_df[insights_df['insight_type'] == 'gap'])
        st.metric("Gaps Identified", gaps)
    with col2:
        synergies = len(insights_df[insights_df['insight_type'] == 'synergy'])
        st.metric("Synergies Found", synergies)
    with col3:
        recommendations = len(insights_df[insights_df['insight_type'] == 'recommendation'])
        st.metric("Recommendations", recommendations)

    st.markdown("---")

    # Display all insights
    for _, insight in insights_df.iterrows():
        card_class = {
            'gap': 'gap-card',
            'synergy': 'synergy-card',
            'recommendation': 'recommendation-card'
        }.get(insight['insight_type'], 'insight-card')

        priority_badge = f"🔴 HIGH" if insight['priority'] == 'high' else "🟡 MEDIUM"
        confidence_badge = f"Confidence: {insight['confidence']:.0%}"

        st.markdown(f"""
        <div class="insight-card {card_class}">
            <h4>{insight['title']}</h4>
            <p><strong>{priority_badge}</strong> | {confidence_badge} | Type: {insight['insight_type'].title()}</p>
            <p>{insight['description']}</p>
        </div>
        """, unsafe_allow_html=True)

def show_insights_by_type(insights_df, insight_type, card_class):
    """Display insights filtered by type"""
    filtered = insights_df[insights_df['insight_type'] == insight_type]

    if len(filtered) == 0:
        st.info(f"No {insight_type}s found in the database.")
        return

    st.subheader(f"{len(filtered)} {insight_type.title()}{'s' if len(filtered) != 1 else ''} Identified")

    for _, insight in filtered.iterrows():
        priority_badge = f"🔴 HIGH PRIORITY" if insight['priority'] == 'high' else "🟡 MEDIUM PRIORITY"
        confidence_badge = f"Confidence: {insight['confidence']:.0%}"

        # Parse entities if it's a JSON string
        entities_text = ""
        try:
            entities = json.loads(insight['entities_involved']) if isinstance(insight['entities_involved'], str) else insight['entities_involved']
            if entities and len(entities) > 0:
                # Show ALL entities, not truncated
                entities_text = f"<p><strong>Entities Involved:</strong> {', '.join(entities)}</p>"
        except:
            pass

        st.markdown(f"""
        <div class="insight-card {card_class}">
            <h4>{insight['title']}</h4>
            <p>{priority_badge} | {confidence_badge}</p>
            <p>{insight['description']}</p>
            {entities_text}
        </div>
        """, unsafe_allow_html=True)

def show_collaboration_finder(df, relationships_df, clusters_df):
    """Display collaboration finder tool"""
    st.header("🤝 Collaboration Finder")

    st.markdown("""
    Find potential collaboration opportunities by searching for entities with complementary
    capabilities, activities, or needs. Explore synergies and partnership possibilities.
    """)

    # Search tabs
    search_tabs = st.tabs(["By Capability", "By Cluster", "Top Synergies"])

    with search_tabs[0]:
        show_capability_search(df, clusters_df)

    with search_tabs[1]:
        show_cluster_collaboration(clusters_df)

    with search_tabs[2]:
        show_top_synergies(relationships_df)

def show_capability_search(df, clusters_df):
    """Search entities by capability"""
    st.subheader("Search by Capability or Activity")

    # Extract all unique CE activities from the dataset
    all_activities = set()
    for activities_str in df['ce_activities'].dropna():
        try:
            # Parse the string representation of the list
            activities_list = ast.literal_eval(activities_str) if isinstance(activities_str, str) else activities_str
            if isinstance(activities_list, list):
                all_activities.update(activities_list)
        except:
            pass

    # Sort activities alphabetically
    sorted_activities = sorted(list(all_activities))

    # Multi-select dropdown
    selected_activities = st.multiselect(
        "Select CE Activities",
        options=sorted_activities,
        default=[],
        help="Select one or more CE activities to find matching entities"
    )

    if selected_activities:
        # Filter entities that have ANY of the selected activities
        matching_entities = df[df['ce_activities'].apply(
            lambda x: any(activity in str(x) for activity in selected_activities) if pd.notna(x) else False
        )]

        st.success(f"Found {len(matching_entities)} entities with selected activities")

        if len(matching_entities) > 0:
            # Display as expandable cards with full descriptions
            for _, entity in matching_entities.head(50).iterrows():
                with st.expander(f"**{entity['entity_name']}** | {entity['ecosystem_role']}"):
                    st.markdown(f"**Ecosystem Role:** {entity['ecosystem_role']}")

                    if pd.notna(entity['brief_description']):
                        st.markdown(f"**Description:**")
                        st.write(entity['brief_description'])  # Full description, not truncated
                    else:
                        st.info("No description available")

                    st.markdown(f"**CE Activities Count:** {entity.get('ce_activities_count', 0)}")

                    if pd.notna(entity.get('ce_activities')):
                        st.markdown(f"**CE Activities:**")
                        st.write(entity['ce_activities'])  # Full activities list

                    if pd.notna(entity.get('latitude')) and pd.notna(entity.get('longitude')):
                        st.markdown(f"**Location:** Lat {entity['latitude']:.6f}, Lon {entity['longitude']:.6f}")

def show_cluster_collaboration(clusters_df):
    """Show collaboration opportunities within clusters with entity lists"""
    st.subheader("Explore Clusters")

    # Cluster type filter
    cluster_type = st.selectbox(
        "Cluster Type",
        options=['All'] + list(clusters_df['cluster_type'].unique())
    )

    filtered_clusters = clusters_df if cluster_type == 'All' else clusters_df[clusters_df['cluster_type'] == cluster_type]

    st.metric("Clusters", len(filtered_clusters))

    # Cluster selection dropdown
    cluster_names = ['All Clusters'] + sorted(filtered_clusters['cluster_name'].tolist())
    selected_cluster = st.selectbox(
        "🔍 Select a cluster",
        options=cluster_names,
        help="Choose a cluster to view its details and entities"
    )

    # Display clusters
    display_count = 0
    for _, cluster in filtered_clusters.iterrows():
        try:
            entities = json.loads(cluster['entities']) if isinstance(cluster['entities'], str) else cluster['entities']
            entity_count = len(entities) if entities else 0
        except:
            entities = []
            entity_count = 0

        # Apply cluster selection filter
        if selected_cluster != 'All Clusters' and cluster['cluster_name'] != selected_cluster:
            continue

        display_count += 1

        # Limit display
        if display_count > 50:
            st.info("Showing first 50 clusters. Use search to find more specific clusters.")
            break

        cluster_color = {
            'capability': 'synergy-card',
            'activity': 'recommendation-card',
            'need': 'gap-card'
        }.get(cluster['cluster_type'], 'insight-card')

        # Use expander for cluster with entity list
        with st.expander(
            f"**{cluster['cluster_name']}** | {cluster['cluster_type'].title()} | {entity_count} entities | {cluster['confidence']:.0%} confidence",
            expanded=False
        ):
            st.markdown(f"**Description:**")
            st.write(cluster['description'])  # Full description, not truncated

            st.markdown(f"**Type:** {cluster['cluster_type'].title()}")
            st.markdown(f"**Confidence:** {cluster['confidence']:.0%}")

            # Display entity list
            if entities and len(entities) > 0:
                st.markdown(f"**Entities in this cluster ({len(entities)}):**")

                # For small clusters, show all as bullet list
                if len(entities) <= 20:
                    for entity in sorted(entities):
                        st.markdown(f"- {entity}")
                else:
                    # For large clusters, show in columns
                    st.markdown(f"*Showing {min(len(entities), 50)} of {len(entities)} entities*")

                    # Create searchable list for large clusters
                    entity_search = st.text_input(
                        f"Search within {cluster['cluster_name']}",
                        key=f"search_{cluster['cluster_id']}",
                        placeholder="Filter entities..."
                    )

                    display_entities = entities
                    if entity_search:
                        display_entities = [e for e in entities if entity_search.lower() in str(e).lower()]

                    # Display in columns for better readability
                    cols = st.columns(2)
                    for idx, entity in enumerate(sorted(display_entities[:50])):
                        cols[idx % 2].markdown(f"- {entity}")

                    if len(display_entities) > 50:
                        st.info(f"+ {len(display_entities) - 50} more entities. Use search to filter.")
            else:
                st.info("No entities in this cluster.")

def show_top_synergies(relationships_df):
    """Show top potential synergies"""
    st.subheader("Top Potential Synergies")

    # Filter for potential synergies
    synergies = relationships_df[relationships_df['relationship_type'] == 'potential_synergy']
    top_synergies = synergies.nlargest(30, 'confidence')

    st.metric("Total Potential Synergies", f"{len(synergies):,}")

    # Display top synergies
    for _, syn in top_synergies.iterrows():
        st.markdown(f"""
        <div class="insight-card synergy-card">
            <h5>{syn['source_entity']} ↔ {syn['target_entity']}</h5>
            <p><strong>Confidence:</strong> {syn['confidence']:.0%}</p>
            <p>{syn['evidence']}</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

