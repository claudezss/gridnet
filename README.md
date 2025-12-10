# Power Grid Model Control Panel (MCP)

A Streamlit-based web application for power system analysis using pandapower, integrated with Ollama's Qwen 32B model for intelligent analysis.

## Features

- Interactive network model selection
- Switch status configuration
- Load profile scaling
- Power flow analysis
- Visual network topology display
- Detailed power flow results and analysis
- AI-powered network analysis and recommendations using Qwen 32B
- FastMCP integration for improved performance

## Prerequisites

1. Install Ollama:
```bash
curl https://ollama.ai/install.sh | sh
```

2. Pull the Qwen 32B model:
```bash
ollama pull qwen:32b
```

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd gridnet
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the MCP application (this will start both the backend and frontend):
```bash
python start_mcp.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Using the MCP:
   - Select a network model from the sidebar
   - Configure switch statuses and load profiles
   - Run power flow analysis
   - Ask AI for analysis and recommendations
   - View results in both graphical and tabular format

## AI Features

The integrated Qwen 32B model provides:
- Network stability analysis
- Performance optimization suggestions
- Potential issue identification
- Load balancing recommendations
- Security assessment
- Custom analysis based on user queries

## Sample Network

The application comes with a sample network that includes:
- 4 buses (110 kV)
- 3 transmission lines
- 2 switches
- 3 loads
- 1 generator
- 1 external grid connection

## Requirements

- Python 3.7+
- pandapower
- streamlit
- plotly
- fastmcp
- Ollama with Qwen 32B model
- Other dependencies listed in requirements.txt

## Architecture

The application consists of three main components:
1. Streamlit frontend for user interaction
2. FastAPI backend for network analysis
3. Ollama integration for AI-powered insights

## Note

Make sure the Ollama service is running and the Qwen 32B model is available before starting the application. The AI analysis features require a system with sufficient computational resources to run the large language model.
