# Grace Unified Orb Interface

A comprehensive AI interface system implementing the Grace Unified Orb Interface specification with advanced reasoning, visual development environment, and multi-layered governance.

## 🌟 Features

### 🧠 Grace Intelligence
- **5-stage reasoning cycle**: Interpretation → Planning → Action → Verification → Response
- **Multi-agent collaboration**: Cross-domain workflows between Trading, Sales, Learning, etc.
- **Dynamic knowledge ingestion**: Real-time feeds, batch documents, user feedback learning
- **Adaptable models**: Context-aware model switching for optimal performance
- **Constitutional compliance**: Built-in trust and ethics verification

### 🛠️ Grace IDE - Live Development Environment
- **Visual flow editor**: Drag-and-drop block-card interface
- **Block registry**: 8+ specialized blocks (API calls, transformations, analysis, etc.)
- **Template library**: Pre-built patterns for common workflows
- **Live validation**: Real-time connection checking and cycle detection
- **Sandbox execution**: Safe testing with trust evaluation and monitoring

### 🔮 Unified Orb Interface
- **Persistent chat**: Memory-enabled conversations with reasoning traces
- **Dynamic panels**: Up to 6 concurrent specialized panels
- **Memory management**: Document upload, indexing, and intelligent search
- **Governance integration**: Approval workflows and audit trails
- **Proactive notifications**: Priority-based alerts with action buttons
- **Multi-modal support**: Text, documents, audio, video processing

### 🏛️ Multi-Layer Governance
- **Layer-1 Constitutional**: Immutable rules (privacy, harm prevention, legal compliance)
- **Layer-2 Organizational**: Configurable policies with override mechanisms
- **Real-time evaluation**: Every action checked against governance rules
- **Guided workflows**: Users directed through compliant processes

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- FastAPI dependencies (see `requirements.txt`)

### Installation
```bash
git clone https://github.com/aaron031291/Grace-.git
cd Grace-
pip install -r requirements.txt
```

### Running Tests
```bash
# Test core functionality
python test_orb_interface.py

# Test governance examples
python governance_examples.py
```

### Starting the API Server
```bash
# Method 1: Direct FastAPI
python -m grace.interface.orb_api

# Method 2: Using the demo (includes sample data)
python demo_orb_interface.py
```

### API Documentation
Once the server is running, visit:
- **Interactive docs**: http://localhost:8080/docs
- **OpenAPI spec**: http://localhost:8080/openapi.json
- **Root endpoint**: http://localhost:8080/

## 📚 API Endpoints

### Sessions
- `POST /api/orb/v1/sessions/create` - Create new session
- `GET /api/orb/v1/sessions/{session_id}` - Get session info
- `DELETE /api/orb/v1/sessions/{session_id}` - End session

### Chat Interface
- `POST /api/orb/v1/chat/message` - Send chat message
- `GET /api/orb/v1/chat/{session_id}/history` - Get chat history

### Panel Management
- `POST /api/orb/v1/panels/create` - Create panel
- `GET /api/orb/v1/panels/{session_id}` - List panels
- `PUT /api/orb/v1/panels/update` - Update panel data
- `DELETE /api/orb/v1/panels/{session_id}/{panel_id}` - Close panel

### Memory Management
- `POST /api/orb/v1/memory/upload` - Upload document
- `POST /api/orb/v1/memory/search` - Search memory
- `GET /api/orb/v1/memory/stats` - Memory statistics

### IDE Integration
- `POST /api/orb/v1/ide/panels/{session_id}` - Open IDE panel
- `POST /api/orb/v1/ide/flows` - Create flow
- `GET /api/orb/v1/ide/flows/{flow_id}` - Get flow details
- `POST /api/orb/v1/ide/flows/blocks` - Add block to flow
- `GET /api/orb/v1/ide/blocks` - Get block registry

### Governance
- `POST /api/orb/v1/governance/tasks` - Create task
- `GET /api/orb/v1/governance/tasks/{user_id}` - Get user tasks
- `PUT /api/orb/v1/governance/tasks/{task_id}/status` - Update task status

### Notifications
- `POST /api/orb/v1/notifications` - Create notification
- `GET /api/orb/v1/notifications/{user_id}` - Get notifications
- `DELETE /api/orb/v1/notifications/{notification_id}` - Dismiss notification

### WebSocket
- `WS /ws/{session_id}` - Real-time communication

## 💡 Usage Examples

### Creating a Session and Chatting
```python
import requests

# Create session
response = requests.post("http://localhost:8080/api/orb/v1/sessions/create", 
                        json={"user_id": "user123"})
session_id = response.json()["session_id"]

# Send chat message
requests.post("http://localhost:8080/api/orb/v1/chat/message",
             json={"session_id": session_id, 
                   "content": "Analyze EUR/USD market data"})

# Get chat history
history = requests.get(f"http://localhost:8080/api/orb/v1/chat/{session_id}/history")
```

### Creating IDE Flows
```python
# Create new flow
flow_response = requests.post("http://localhost:8080/api/orb/v1/ide/flows",
                             json={"name": "Trading Analysis", 
                                   "description": "Automated trading flow",
                                   "creator_id": "user123"})
flow_id = flow_response.json()["flow_id"]

# Add blocks to flow
requests.post("http://localhost:8080/api/orb/v1/ide/flows/blocks",
             json={"flow_id": flow_id,
                   "block_type_id": "api_fetch",
                   "position": {"x": 100, "y": 100}})
```

### Memory Management
```python
# Upload document
with open("document.txt", "rb") as f:
    files = {"file": ("document.txt", f, "text/plain")}
    response = requests.post("http://localhost:8080/api/orb/v1/memory/upload",
                            files=files, data={"user_id": "user123"})

# Search memory
search_results = requests.post("http://localhost:8080/api/orb/v1/memory/search",
                              json={"session_id": session_id, 
                                    "query": "trading strategy"})
```

## 🏗️ Architecture

### Core Components
- **GraceIntelligence**: 5-stage reasoning engine with constitutional compliance
- **GraceIDE**: Visual flow editor with sandbox execution
- **GraceUnifiedOrbInterface**: Main orchestrator for all functionality
- **GraceGovernanceEngine**: Multi-layer governance with approval workflows

### Data Flow
1. User input → Grace Intelligence (reasoning)
2. Actions → Governance evaluation (compliance)
3. Approved actions → Domain pods (execution)  
4. Results → UI panels (visualization)
5. Context → Memory storage (learning)

### Panel Types
- **Chat**: Persistent conversation interface
- **Trading**: Financial analysis and execution
- **Sales**: CRM and campaign management
- **Analytics**: Data visualization and reporting
- **Memory**: Knowledge search and management
- **Governance**: Approval and audit workflows
- **IDE**: Visual development environment

## 🔒 Governance & Security

### Layer-1 Constitutional Rules (Immutable)
- **Privacy Protection**: No PII exposure
- **Harm Prevention**: Risk assessment and mitigation
- **Legal Compliance**: Regulatory adherence
- **Transparency**: Audit trails and explainability
- **Fairness**: Bias detection and prevention

### Layer-2 Organizational Policies (Configurable)
- **Trading Limits**: Risk exposure controls
- **Data Access**: Cross-department restrictions
- **Marketing Timing**: Campaign schedule rules
- **Model Deployment**: Review requirements

### Trust Scoring
- Document trust based on source and verification
- Action trust based on compliance and outcomes
- Real-time trust evaluation for all operations

## 🧪 Testing

### Automated Tests
```bash
# Core functionality tests
python test_orb_interface.py

# Governance workflow tests  
python governance_examples.py

# API integration demo
python demo_orb_interface.py
```

### Manual Testing
1. Start the API server
2. Visit http://localhost:8080/docs
3. Use the interactive API documentation
4. Test WebSocket connections
5. Upload documents and test memory search

## 📊 System Statistics

The system provides comprehensive statistics via `/api/orb/v1/stats`:

- **Sessions**: Active sessions and message counts
- **Memory**: Fragment counts and trust scores
- **Governance**: Task status and approval rates
- **IDE**: Flow creation and block usage
- **Intelligence**: Domain pod utilization

## 🔧 Configuration

### Grace Intelligence
- Domain pods: Trading, Sales, Learning, Governance, Development, Analytics
- Model registry: Summarization, forecasting, sentiment, code analysis
- Constitutional rules: Configurable thresholds and checks

### Panel System
- Maximum 6 panels per session
- Configurable panel templates
- Auto-positioning and sizing

### Governance Engine
- Layer-1 rules (immutable)
- Layer-2 policies (configurable with overrides)
- Approval workflows with role-based access

## 🚀 Deployment

### Development
```bash
python -m grace.interface.orb_api
```

### Production
```bash
uvicorn grace.interface.orb_api:app --host 0.0.0.0 --port 8080 --workers 4
```

### Docker (Future)
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "grace.interface.orb_api:app", "--host", "0.0.0.0", "--port", "8080"]
```

## 📝 Development

### Project Structure
```
Grace-/
├── grace/
│   ├── intelligence/           # Grace Intelligence reasoning
│   │   ├── grace_intelligence.py
│   │   └── __init__.py
│   └── interface/             # Orb Interface components
│       ├── ide/               # IDE components
│       │   ├── grace_ide.py
│       │   └── __init__.py
│       ├── orb_interface.py   # Main orb interface
│       ├── orb_api.py         # FastAPI service
│       └── interface_service.py  # Existing interface
├── test_orb_interface.py      # Core functionality tests
├── governance_examples.py     # Governance demonstrations
├── demo_orb_interface.py      # API demo script
└── requirements.txt           # Dependencies
```

### Key Classes
- `GraceIntelligence`: Main reasoning engine
- `GraceIDE`: Visual development environment
- `GraceUnifiedOrbInterface`: Orchestrating interface
- `GraceGovernanceEngine`: Multi-layer governance

### Extension Points
- **New block types**: Extend `BlockType` enum and block registry
- **Custom panels**: Add new `PanelType` values and templates
- **Domain pods**: Extend intelligence domain pod registry
- **Governance rules**: Add Layer-2 policies or enhance Layer-1 rules

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Ensure governance compliance
5. Update documentation
6. Submit a pull request

## 📄 License

This project implements governance principles of transparency and democratic oversight.

---

**Grace Unified Orb Interface** - Where AI reasoning meets human governance 🔮✨