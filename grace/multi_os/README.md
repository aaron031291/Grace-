# Multi-OS Kernel

A unified execution layer across Linux, Windows, and macOS with sandboxing, RBAC, package/runtime setup, blue/green agent upgrades, and image/snapshot rollback capabilities.

## 🎯 Purpose

The Multi-OS Kernel provides:

- **Unified execution layer** across Linux/Windows/macOS (+ container/VM/remote)
- **Normalized operations** for process, filesystem, networking, GPU, and package management behind one API
- **Security & governance** enforcement (RBAC, sandbox, signing, network policy) with rich telemetry
- **Snapshot management** with golden images, blue/green agent rollout, and instant rollback

## 🏗️ Architecture

```
multi_os/
├── agents/              # Per-OS agents (linux, windows, macos, container, vm)
├── adapters/            # Capability adapters (process, fs, net, gpu, pkg)
├── runtime/             # Python/Conda/Venv/Node/Java toolchains
├── sandbox/             # Namespace/jail/AppContainer/SIP wrappers
├── orchestrator/        # Scheduler, placement, blue/green, health
├── inventory/           # Hosts, capabilities, labels, constraints
├── secrets/             # Vault handles, token rotation
├── policy/              # RBAC, egress/ingress, signing
├── telemetry/           # Metrics, logs, traces
├── snapshots/           # Agent + image snapshots, rollbacks
├── bridges/             # Integration bridges (mesh, gov, mlt, memory)
├── contracts/           # JSON/YAML schemas and API specs
└── multi_os_service.py  # FastAPI façade
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install fastapi uvicorn pydantic websockets

# Clone the repository
git clone <repository-url>
cd Grace-
```

### Basic Usage

```python
from multi_os.multi_os_service import MultiOSService

# Initialize the service
service = MultiOSService()

# Register a host
host_descriptor = {
    "host_id": "my-linux-host",
    "os": "linux",
    "arch": "x86_64",
    "agent_version": "2.4.1",
    "capabilities": ["process", "fs", "net", "pkg", "sandbox"],
    "labels": ["region:us-west", "gpu:nvidia"],
    "status": "online"
}

host_id = service.registry.register_host(host_descriptor)

# Submit a task
task = {
    "task_id": "my-task-001",
    "command": "python",
    "args": ["--version"],
    "runtime": {"runtime": "python", "version": "3.11"},
    "constraints": {
        "os": ["linux", "macos"],
        "sandbox": "nsjail"
    }
}

hosts = service.registry.list_hosts({"status": "online"})
placement = await service.scheduler.place(task, hosts)
print(f"Task placed on: {placement['host_id']}")
```

### Running the Demo

```bash
# Run the demonstration
python demo_multi_os_kernel.py

# Run comprehensive tests
python test_multi_os_kernel.py
```

## 📋 Core Capabilities

### 🖥️ Multi-OS Support

- **Linux**: subprocess, cgroups/namespace sandbox (nsjail/firejail), apt/yum/dnf, systemd, Docker/Podman, NVIDIA CUDA
- **Windows**: CreateProcess/AppContainer, Win32/PowerShell, winget/choco/MSI, Windows Defender/WDAC, Hyper-V/WSL, DirectML/CUDA
- **macOS**: posix_spawn, Sandbox/Entitlements, Homebrew/pkgutil, launchd, Apple Silicon GPU/Metal
- **Container**: Docker/OCI run, image pull/build, filesystem bind-mounts, network policy per namespace
- **VM**: QCOW/AMI runners, snapshot/restore, cloud-init

### 🎯 Intelligent Placement

The scheduler uses 4-factor optimization:

- **Capability Fit (40%)**: How well host capabilities match task requirements
- **Latency (25%)**: Geographic distance, network latency, historical performance
- **Success Rate (25%)**: Historical task completion rate, error frequency
- **GPU Availability (10%)**: GPU type and availability scoring

### 📊 Telemetry & KPIs

Tracks comprehensive metrics:

- **Reliability**: placement success %, task failure %, MTTR
- **Performance**: cold-start ms, p95 task latency, GPU util, cache hit rate
- **Security**: sandbox violations, unsigned bundle attempts, denied egress
- **Operations**: rollout duration, rollback count, image drift events, host health SLO

### 📸 Snapshots & Rollback

Multi-scope snapshot support:

- **Agent**: agent versions, runtime caches, sandbox profiles, placement weights
- **Image**: golden images, base configurations, security policies
- **VM**: VM templates, network config, storage config
- **Container**: container images, configs, registry settings

Blue/green rollback with:
- Freeze task placements (drain mode)
- Switch agent to previous version
- Restore configurations and policies
- Verify health and resume operations

## 🔌 REST API

Base URL: `/api/mos/v1`

### Core Endpoints

```http
GET  /health                    # Health check
GET  /hosts                     # List registered hosts
POST /hosts/register            # Register new host
POST /task/submit               # Submit task for execution
GET  /task/{task_id}/status     # Get task status
POST /fs                        # Execute filesystem action
POST /net                       # Execute network action
POST /runtime/ensure            # Ensure runtime environment
POST /agent/rollout             # Start agent rollout
POST /snapshot/export           # Export system snapshot
POST /rollback                  # Rollback to snapshot
```

### Monitoring Endpoints

```http
GET  /metrics                   # Get telemetry metrics
GET  /events                    # Get recent events
GET  /placement/stats           # Get scheduler statistics
```

## 🔒 Security & Governance

- **RBAC**: Actions checked against "mos.*" permissions (mos.task.submit, mos.fs.write, mos.agent.rollout)
- **Sandboxing**: Per-task via constraints.sandbox; default deny-by-default network with allowlist
- **Signing**: Agent updates and task bundles must be signature-verified
- **Secrets**: All tokens/keys referenced as secrets_ref (vault only; never plaintext)
- **Audit**: Every action/event logged to Immutable Logs with host, user, hash

## 📝 Configuration

Default configuration example:

```yaml
multi_os:
  placement:
    weights: {capability_fit: 0.4, latency: 0.25, success: 0.25, gpu: 0.1}
  sandbox:
    default: "nsjail"
    profiles: {linux: "ns_v5", windows: "appcontainer_low", macos: "sandboxd_v3"}
  network:
    default_policy: "deny_all"
    allowlist: ["api.company.local"]
  runtimes:
    prewarm: ["python@3.11", "node@18"]
  rollout:
    strategy: "blue_green"
    rings: ["canary:5%", "ring1:25%", "ring2:50%", "ring3:100%"]
  timeouts:
    task_max_runtime_s: 1800
```

## 🧪 Testing

The Multi-OS Kernel includes comprehensive testing:

```bash
# Run all tests
python test_multi_os_kernel.py

# Results:
✅ Service Initialized: Multi-OS Service with 3 OS adapters
✅ Host Management: Multi-host registration across all OS types
✅ Task Scheduling: Intelligent placement with 4-factor optimization
✅ Telemetry System: 10 KPIs tracked with comprehensive metrics
✅ Snapshot Management: Full snapshot/rollback capabilities
✅ Event Mesh: Event publishing with routing rules
✅ Multi-OS Support: Linux, Windows, macOS adapters with unified API
✅ Security & Governance: Sandboxing, RBAC, policy enforcement ready
✅ API Contract: OpenAPI 3.0 specification with full validation
```

## 🤝 Integration

### Event Mesh Integration

The Multi-OS Kernel publishes events for integration:

- `MOS_HOST_REGISTERED`: Host registration events
- `MOS_TASK_SUBMITTED/STARTED/COMPLETED`: Task lifecycle events
- `MOS_HOST_HEALTH`: Host health updates
- `MOS_AGENT_ROLLING_UPDATE`: Agent rollout progress
- `MOS_SNAPSHOT_CREATED/ROLLBACK_*`: Snapshot and rollback events
- `MOS_EXPERIENCE`: Experience data for meta-learning

### Governance Integration

- Permission checking for task execution
- Policy validation for snapshot operations
- Audit trail for all operations

### MLT Integration

- Experience data collection
- Optimization suggestions consumption
- Adaptive placement weight tuning

## 🏆 Key Benefits

- **🎯 Unified API**: Single API across all operating systems
- **⚡ High Performance**: Intelligent placement with sub-second decisions
- **🔒 Secure**: Multi-layer security with sandboxing and RBAC
- **📊 Observable**: Rich telemetry with 10 tracked KPIs
- **🔄 Reliable**: Blue/green deployments with instant rollback
- **🌐 Scalable**: Designed for multi-region, multi-cloud deployments

## 📄 License

This project is part of the Grace ML platform.

## 🤝 Contributing

Please follow the existing code patterns and ensure all tests pass:

```bash
python test_multi_os_kernel.py
```

---

**Multi-OS Kernel** - Unified execution across all platforms with enterprise-grade reliability and security.