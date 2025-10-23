"""
Grace AI - Refactored 4-Layer Architecture
==========================================

L0: RUNTIME/INFRA
    Containers, queues, storage, runners
    grace/infra/
        ├── containers.py
        ├── queues.py
        ├── storage.py
        └── runners.py

L1: TRUTH & AUDIT (Source of Truth)
    Immutable logging, crypto keys, KPIs, trust, health, clarity
    grace/truth/
        ├── immutable_logger.py       (Audit trail)
        ├── crypto_keys.py            (Cryptographic signing)
        ├── system_metrics.py         (KPIs, health)
        ├── trust_ledger.py           (Trust scores)
        └── clarity_logs.py           (Clarity framework logging)

L2: ORCHESTRATION (TriggerMesh/Event Bus)
    Routes events → actions; loads/validates workflows; fan-in/out
    grace/orchestration/
        ├── trigger_mesh.py           (Event router)
        ├── event_bus.py              (Message bus)
        ├── workflows.py              (Workflow definitions)
        └── validators.py             (Workflow validation)

L3: CONTROL & INTELLIGENCE (Peer layers)
    
    A) MetaKernel (decide_and_route, prioritise, arbitrate)
       grace/meta_kernel/
           ├── meta_learning_kernel.py
           ├── decision_engine.py
           └── arbitrator.py
    
    B) MCP (Model Context Protocol)
       grace/mcp/
           ├── manager.py
           ├── protocol.py
           ├── context_adapters.py
           ├── tool_registry.py
           └── discovery.py
    
    C) Transcendence (Self-improvement)
       grace/transcendence/
           ├── mentor_engine.py
           ├── sandbox_manager.py
           ├── code_learning_engine.py
           └── improvement_cycles.py

L4: EXECUTORS/SERVICES
    Remote agent, workflow handlers, analyzers, CI, tests, deployers
    grace/executors/
        ├── remote_agent.py
        ├── workflow_handlers.py
        ├── analyzers.py
        ├── ci_cd.py
        ├── test_runner.py
        └── deployer.py

INTERFACES
    grace/api/
        ├── rest_api.py
        └── websocket_api.py
    
    grace/frontend/
        ├── index.html
        ├── style.css
        └── app.js
"""
