#!/usr/bin/env python3
"""
CLI for managing TriggerMesh workflows.

Provides commands to list, inspect, enable, disable, and reload workflows.
"""
import argparse
import sys
from pathlib import Path

# Add project root to path to allow importing grace modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from grace.orchestration.workflow_registry import WorkflowRegistry


def list_workflows(registry: WorkflowRegistry):
    """List all loaded workflows."""
    workflows = registry.get_all_workflows()
    if not workflows:
        print("No workflows loaded.")
        return

    print(f"{'NAME':<35} {'ENABLED':<10} {'TRIGGER EVENT':<30}")
    print("-" * 75)
    for wf in sorted(workflows, key=lambda w: w["name"]):
        name = wf["name"]
        enabled = "✅" if wf.get("enabled", True) else "❌"
        trigger = wf["trigger_event"]
        print(f"{name:<35} {enabled:<10} {trigger:<30}")


def show_workflow(registry: WorkflowRegistry, name: str):
    """Show details of a specific workflow."""
    wf = registry.get_workflow_by_name(name)
    if not wf:
        print(f"Error: Workflow '{name}' not found.")
        sys.exit(1)

    print(f"Name:        {wf['name']}")
    print(f"Description: {wf.get('description', 'N/A')}")
    print(f"Enabled:     {wf.get('enabled', True)}")
    print(f"Trigger:     {wf['trigger_event']}")
    print("\nFilters:")
    for f in wf.get("filters", []):
        print(f"  - {f['field']} {f['operator']} {f['value']}")
    print("\nActions:")
    for a in wf.get("actions", []):
        print(f"  - Name:   {a['name']}")
        print(f"    Target: {a['target']}")
        print(f"    Params: {a.get('params', {})}")


def reload_workflows(registry: WorkflowRegistry):
    """Hot-reload all workflows from disk."""
    print("Reloading workflows...")
    registry.load_workflows()
    stats = registry.get_stats()
    print(f"✅ Reload complete. Loaded {stats['workflows_loaded']} workflows.")
    if stats["validation_errors"] > 0:
        print(f"⚠️ Found {stats['validation_errors']} validation errors.")


def validate_workflows(registry: WorkflowRegistry):
    """Validate all workflow files."""
    print("Validating workflows...")
    registry.load_workflows() # load_workflows runs validation
    stats = registry.get_stats()
    if stats["validation_errors"] == 0:
        print(f"✅ All {stats['workflows_loaded']} workflows are valid.")
    else:
        print(f"❌ Found {stats['validation_errors']} validation errors.")
        sys.exit(1)


def show_stats(registry: WorkflowRegistry):
    """Show loading statistics."""
    stats = registry.get_stats()
    print("Workflow Registry Stats:")
    print(f"  - Workflows Loaded: {stats['workflows_loaded']}")
    print(f"  - Validation Errors: {stats['validation_errors']}")
    trigger_types = registry.get_all_trigger_event_types()
    print(f"  - Unique Trigger Events: {len(trigger_types)}")


def main():
    parser = argparse.ArgumentParser(description="Manage TriggerMesh Workflows")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List all workflows")
    show_parser = subparsers.add_parser("show", help="Show details for a workflow")
    show_parser.add_argument("name", help="Name of the workflow")
    subparsers.add_parser("reload", help="Hot-reload all workflows from disk")
    subparsers.add_parser("validate", help="Validate all workflow definitions")
    subparsers.add_parser("stats", help="Show registry statistics")
    
    # Mock commands for enable/disable as they require runtime state
    enable_parser = subparsers.add_parser("enable", help="Enable a workflow (requires runtime)")
    enable_parser.add_argument("name", help="Name of the workflow")
    disable_parser = subparsers.add_parser("disable", help="Disable a workflow (requires runtime)")
    disable_parser.add_argument("name", help="Name of the workflow")

    args = parser.parse_args()

    workflow_dir = Path(__file__).parent.parent / "grace" / "orchestration" / "workflows"
    registry = WorkflowRegistry(workflow_dir)
    registry.load_workflows()

    if args.command == "list":
        list_workflows(registry)
    elif args.command == "show":
        show_workflow(registry, args.name)
    elif args.command == "reload":
        reload_workflows(registry)
    elif args.command == "validate":
        validate_workflows(registry)
    elif args.command == "stats":
        show_stats(registry)
    elif args.command in ["enable", "disable"]:
        print(f"'{args.command}' is a runtime operation. This CLI only confirms workflow existence.")
        if registry.get_workflow_by_name(args.name):
            print(f"✅ Workflow '{args.name}' found.")
        else:
            print(f"❌ Workflow '{args.name}' not found.")


if __name__ == "__main__":
    main()
