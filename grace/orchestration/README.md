# Grace TriggerMesh Orchestration Layer

This directory contains the core components of the TriggerMesh, an event-driven orchestration layer for Grace. It connects different parts of the system (kernels, databases, APIs) through declarative, event-driven workflows.

## 🧠 Core Principles

1.  **Repository Alignment**: Workflows are defined in version-controlled YAML files (`workflows/`), not in code. This makes them easy to review, test, and manage.
2.  **Schema-Aware**: All events are defined with a clear schema (`schemas/event_schemas.yaml`), ensuring data consistency and enabling validation.
3.  **Cohesion with Existing Architecture**: TriggerMesh integrates with and enhances existing components like the `EventBus`, `ImmutableLogger`, and `KPITrustMonitor`. It does not replace them.
4.  **Separation of Concerns**:
    *   **Orchestration (`orchestration/`)**: Defines *what* should happen in response to an event.
    *   **Kernels (`kernels/`)**: Define *how* to perform a specific action. TriggerMesh routes events to the correct kernel.

## 📂 Directory Structure

```
orchestration/
├── README.md                      # This architecture guide
├── __init__.py                    # Exports core components
├── event_router.py                # Routes events to workflows
├── workflow_engine.py             # Executes workflow actions
├── workflow_registry.py           # Loads and manages workflows from YAML
└── workflows/
    └── trigger_mesh_workflows.yaml  # Declarative, event-driven workflows
```

## ⚙️ How It Works

The TriggerMesh operates in a simple, powerful loop:

1.  **Listen**: The `EventRouter` subscribes to topics on the main `EventBus`.
2.  **Route**: When an event is received, the `EventRouter` checks the `WorkflowRegistry` for any workflows that are triggered by that event type.
3.  **Filter**: It applies any filters defined on the workflow (e.g., `severity == "CRITICAL"`) to decide if the workflow should run. It also handles rate-limiting and deduplication.
4.  **Execute**: If the filters pass, the `WorkflowEngine` takes over and executes the actions defined in the workflow.
5.  **Act**: The `WorkflowEngine` calls the appropriate registered **kernel handler** to perform the work (e.g., `avn_core.escalate_healing`).
6.  **Log**: The entire process, from event receipt to action completion, is logged in the `ImmutableLogger` for full traceability.

### Event Flow Diagram

```
┌─────────────────┐
│  System Event   │ (e.g., KPI breach, DB update)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    EventBus     │
└────────┬────────┘
         │ (Subscribed)
         ▼
┌─────────────────┐
│  EventRouter    │
└────────┬────────┘
         │ 1. Find matching workflow
         │ 2. Check filters & rate limits
         ▼
┌─────────────────┐
│ WorkflowEngine  │
└────────┬────────┘
         │ 3. Execute actions
         ▼
┌─────────────────┐
│ Kernel Handlers │ (e.g., AVN Core, Learning Kernel)
└────────┬────────┘
         │ 4. Perform business logic
         ▼
┌─────────────────┐
│ ImmutableLogs   │ (Full audit trail)
└─────────────────┘
```

## 🚀 Getting Started

### 1. Define a Workflow

Add a workflow to `workflows/trigger_mesh_workflows.yaml`:

```yaml
- name: my_new_workflow
  description: "A new workflow for handling a custom event."
  trigger_event: "my_app.custom_event"
  enabled: true
  filters:
    - field: "priority"
      operator: "=="
      value: "high"
  actions:
    - name: "Take Action"
      target: "my_kernel.take_action"
      params:
        item_id: "{{ payload.item_id }}"
```

### 2. Register a Kernel Handler

In your application's setup code, register the function that performs the action:

```python
from my_app.kernels import my_kernel
# ...
workflow_engine.register_kernel_handler("my_kernel.take_action", my_kernel.take_action)
```

### 3. Publish an Event

Publish an event to the `EventBus`. The `EventRouter` will automatically pick it up and trigger your workflow.

```python
await event_bus.publish(
    "my_app.custom_event",
    {"priority": "high", "item_id": "xyz-123"}
)
```

## 🛠️ Management & Demo

-   **CLI**: Use the command-line tool to manage and inspect workflows:
    ```bash
    python scripts/manage_workflows.py --help
    ```

-   **Demo**: Run the integration demo to see the TriggerMesh in action:
    ```bash
    python scripts/demo_triggermesh.py
    ```

## ✅ Benefits

-   **Decoupling**: Logic is decoupled from orchestration. Kernels don't need to know what triggers them.
-   **Agility**: Change orchestration logic by editing a YAML file, not by deploying new code.
-   **Traceability**: Every event and action is logged immutably.
-   **Resilience**: Features like rate-limiting and timeouts prevent system overload.
-   **Clarity**: The state of the system's logic is readable and centrally located.
