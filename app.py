import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ✅ Page configuration with enhanced dark theme and sleek UI
st.set_page_config(
    page_title="DataViz Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/AryanAgarwal27',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': 'DataViz Pro - AI-powered interactive data visualization'
    }
)

# Reset to light theme (default Streamlit look)
# Remove dark theme styling

# Sidebar with updated branding and a fresh, relevant image
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3826/3826579.png", width=120)  # Fresh, relevant data visualization image
    st.title("DataViz Pro")

    with st.expander("ℹ️ About", expanded=False):
        st.markdown("""
        **DataViz Pro** empowers you to analyze and visualize CSV data using AI.
        - 📊 Smart and intuitive data visualizations
        - 💬 AI-driven insights and recommendations
        - 🚀 Fast and interactive data exploration
        """)

    with st.expander("📝 History", expanded=True):
        st.subheader("Recent Queries")
        if "history" in st.session_state and st.session_state.history:
            for q, a in st.session_state.history[-5:]:
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a[:100]}...")
                st.divider()
        else:
            st.info("No queries yet. Start by asking a question!")

    with st.expander("⚙️ Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("☀️ **Light mode enabled**")
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.rerun()

st.title("📊 AI-Powered Data Visualization")


# Initialize session states
if "df" not in st.session_state:
    st.session_state.df = None
if "history" not in st.session_state:
    st.session_state.history = []

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found. Check your .env file.")
    st.stop()

# Custom prompt template for the agent
CUSTOM_PROMPT = """You are a data analysis assistant. When asked to create visualizations, ALWAYS use Plotly Express (px) instead of matplotlib or seaborn. 
Also, try to make the visualization as beautiful as possible. Don't make it blatant by using simple colors.
"""

# Common plotting functions
def plot_value_counts(df, column, title=None, xlabel=None, ylabel="Count"):
    # Create value counts
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = [column, 'Count']
    
    # Create interactive bar plot
    fig = px.bar(value_counts, 
                 x=column, 
                 y='Count',
                 title=title or f"Distribution of {column}",
                 labels={column: xlabel or column, 'Count': ylabel},
                 template="plotly_white")  # Using a clean template
    
    # Update layout for better appearance and inline display
    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-45,
        height=500,
        width=None,  # Let Streamlit control the width
        margin=dict(t=50, l=50, r=50, b=50),
        hovermode='x unified',
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    
    return fig

# File Upload Section with better UI
col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
with col2:
    if uploaded_file:
        st.success("File uploaded successfully!")
        
if uploaded_file:
    st.session_state.df = pd.read_csv(uploaded_file)
    with st.expander("📋 Data Preview", expanded=True):
        st.dataframe(st.session_state.df.head(10), use_container_width=True)

# Initialize agent with custom prompt
def create_agent(df):
    prompt = ChatPromptTemplate.from_template(CUSTOM_PROMPT)
    return create_pandas_dataframe_agent(
        llm=ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini"),
        df=df,
        agent_type="openai-tools",
        verbose=True,
        handle_parsing_errors=True,
        allow_dangerous_code=True,
        prefix=CUSTOM_PROMPT
    )

# Function to handle plot generation
def handle_visualization(code_str):
    try:
        # Create a container for the visualization
        viz_container = st.container()
        
        # Execute the code directly if it contains Plotly Express
        if "px." in code_str:
            # Add DataFrame reference
            code_str = code_str.replace("df.", "st.session_state.df.")
            
            # Add template and layout configurations to make plots inline
            if "px.bar" in code_str or "px.line" in code_str or "px.scatter" in code_str:
                code_str = code_str.replace("px.", "px.").replace("fig =", """fig = """)
                if "update_layout" not in code_str:
                    code_str += """
fig.update_layout(
    template="plotly_white",
    height=500,
    margin=dict(t=50, l=50, r=50, b=50),
    hovermode='x unified',
    plot_bgcolor="white",
    paper_bgcolor="white"
)"""
            
            # Execute the code
            local_dict = {"px": px, "pd": pd, "st": st, "df": st.session_state.df}
            exec(code_str, globals(), local_dict)
            
            # If fig was created, display it in the container
            if "fig" in local_dict:
                with viz_container:
                    st.plotly_chart(local_dict["fig"], use_container_width=True, config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'scrollZoom': True
                    })
                return True
            
        # Fallback to our custom plotting for value counts
        if "Model Year" in code_str and ("value_counts" in code_str or "count" in code_str.lower()):
            fig = plot_value_counts(st.session_state.df, "Model Year", 
                                  "Distribution of Vehicles by Model Year",
                                  "Model Year")
            with viz_container:
                st.plotly_chart(fig, use_container_width=True, config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'scrollZoom': True
                })
            return True
            
        return False
    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")
        return False

# Query Processing with improved UI
if st.session_state.df is not None:
    st.markdown("### 💬 Ask about your data")
    query = st.text_input("Enter your question:", placeholder="e.g., 'Show me the distribution of Model Years'")
    
    # Create a container for results
    results_container = st.container()
    
    if query:
        with st.spinner("🤔 Analyzing your data..."):
            try:
                agent = create_agent(st.session_state.df)
                response = agent.invoke({"input": query})
                
                # Check if response contains plotting code or visualization request
                if any(indicator in response["output"].lower() for indicator in 
                      ['plot', 'figure', 'px.', 'bar', 'scatter', 'hist', 'line', 'count', 'distribution']):
                    
                    with st.spinner("📊 Generating visualization..."):
                        with results_container:
                            success = handle_visualization(response["output"])
                            if success:
                                st.success("Chart generated successfully!")
                                st.session_state.history.append((query, "Interactive chart generated successfully"))
                            else:
                                st.write(response["output"])
                                st.session_state.history.append((query, "Failed to generate chart"))
                else:
                    with results_container:
                        st.write(response["output"])
                        st.session_state.history.append((query, response["output"]))
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.history.append((query, f"Error: {str(e)}"))
else:
    st.info("👆 Please upload a CSV file to begin analysis")
