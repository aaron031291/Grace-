#!/usr/bin/env python3
"""
TriggerMesh Workflow Management CLI

Usage:
    python scripts/manage_workflows.py list               # List all workflows
    python scripts/manage_workflows.py show <name>        # Show workflow details
    python scripts/manage_workflows.py enable <name>      # Enable a workflow
    python scripts/manage_workflows.py disable <name>     # Disable a workflow
    python scripts/manage_workflows.py reload             # Hot-reload workflows
    python scripts/manage_workflows.py stats              # Show registry stats
    python scripts/manage_workflows.py validate           # Validate all workflows
"""

import sys
import argparse
from pathlib import Path

# Add Grace root to path
grace_root = Path(__file__).parent.parent
sys.path.insert(0, str(grace_root))

from grace.orchestration.workflow_registry import WorkflowRegistry
import json


def list_workflows(registry: WorkflowRegistry, enabled_only: bool = False):
    """List all workflows."""
    workflows = registry.list_workflows(enabled_only=enabled_only)

    if not workflows:
        print("No workflows found")
        return

    print(f"\n{'=' * 80}")
    print(f"{'NAME':<30} {'STATUS':<10} {'TRIGGER EVENT':<30}")
    print(f"{'=' * 80}")

    for wf in workflows:
        status = "✅ ENABLED" if wf["enabled"] else "❌ DISABLED"
        print(f"{wf['name']:<30} {status:<10} {wf['trigger_event']:<30}")
        if wf["description"]:
            print(f"  └─ {wf['description']}")
        print(f"     Actions: {wf['actions_count']}, Version: {wf['version']}")

    print(f"{'=' * 80}\n")
    print(f"Total workflows: {len(workflows)}")


def show_workflow(registry: WorkflowRegistry, name: str):
    """Show detailed workflow information."""
    workflow = registry.get_workflow(name)

    if not workflow:
        print(f"❌ Workflow '{name}' not found")
        return

    print(f"\n{'=' * 80}")
    print(f"Workflow: {workflow.name}")
    print(f"{'=' * 80}")
    print(f"Description: {workflow.description}")
    print(f"Enabled: {'✅ Yes' if workflow.enabled else '❌ No'}")
    print(f"Version: {workflow.version}")
    print(f"\nTrigger:")
    print(f"  Event Type: {workflow.trigger.event_type}")
    print(f"  Filters: {json.dumps(workflow.trigger.filters, indent=4)}")

    print(f"\nActions ({len(workflow.actions)}):")
    for i, action in enumerate(workflow.actions, 1):
        print(f"\n  {i}. {action.name}")
        print(f"     Target: {action.target_kernel}.{action.action}")
        print(f"     Priority: {action.priority}")
        print(f"     Timeout: {action.timeout_ms}ms")
        if action.parameters:
            print(f"     Parameters: {json.dumps(action.parameters, indent=8)}")

    if workflow.logging:
        print(f"\nLogging:")
        for key, value in workflow.logging.items():
            print(f"  {key}: {value}")

    print(f"{'=' * 80}\n")


def show_stats(registry: WorkflowRegistry):
    """Show registry statistics."""
    stats = registry.get_stats()

    print(f"\n{'=' * 80}")
    print("Workflow Registry Statistics")
    print(f"{'=' * 80}")
    print(f"Total Workflows: {stats['total_workflows']}")
    print(f"Enabled: {stats['enabled_workflows']}")
    print(f"Disabled: {stats['disabled_workflows']}")
    print(f"Unique Event Types: {stats['unique_event_types']}")

    print(f"\nWorkflows by Kernel:")
    for kernel, count in sorted(stats['workflows_by_kernel'].items()):
        print(f"  {kernel}: {count}")

    print(f"{'=' * 80}\n")


def validate_workflows(registry: WorkflowRegistry):
    """Validate all workflow definitions."""
    print("Validating workflows...")

    all_valid = True
    for workflow in registry.workflows:
        # Check basic fields
        if not workflow.name:
            print(f"❌ Workflow missing name")
            all_valid = False
            continue

        # Check trigger
        if not workflow.trigger.event_type:
            print(f"❌ Workflow '{workflow.name}' missing trigger event_type")
            all_valid = False

        # Check actions
        if not workflow.actions:
            print(f"⚠️  Workflow '{workflow.name}' has no actions")

        for action in workflow.actions:
            if not action.target_kernel:
                print(f"❌ Action '{action.name}' in workflow '{workflow.name}' missing target_kernel")
                all_valid = False

            if not action.action:
                print(f"❌ Action '{action.name}' in workflow '{workflow.name}' missing action")
                all_valid = False

    if all_valid:
        print(f"✅ All {len(registry.workflows)} workflows are valid")
    else:
        print(f"❌ Some workflows have validation errors")

    return all_valid


def main():
    parser = argparse.ArgumentParser(
        description="TriggerMesh Workflow Management CLI"
    )
    parser.add_argument(
        "command",
        choices=["list", "show", "enable", "disable", "reload", "stats", "validate"],
        help="Command to execute",
    )
    parser.add_argument("name", nargs="?", help="Workflow name (for show/enable/disable)")
    parser.add_argument(
        "--enabled-only", action="store_true", help="Show only enabled workflows (for list)"
    )

    args = parser.parse_args()

    # Initialize registry
    workflow_dir = grace_root / "grace" / "orchestration" / "workflows"
    registry = WorkflowRegistry()
    registry.load_workflows(str(workflow_dir))

    # Execute command
    if args.command == "list":
        list_workflows(registry, enabled_only=args.enabled_only)

    elif args.command == "show":
        if not args.name:
            print("❌ Error: 'show' command requires workflow name")
            parser.print_help()
            sys.exit(1)
        show_workflow(registry, args.name)

    elif args.command == "enable":
        if not args.name:
            print("❌ Error: 'enable' command requires workflow name")
            sys.exit(1)
        registry.enable_workflow(args.name)
        print(f"✅ Workflow '{args.name}' enabled")

    elif args.command == "disable":
        if not args.name:
            print("❌ Error: 'disable' command requires workflow name")
            sys.exit(1)
        registry.disable_workflow(args.name)
        print(f"✅ Workflow '{args.name}' disabled")

    elif args.command == "reload":
        registry.reload_workflows(str(workflow_dir))
        print(f"✅ Workflows reloaded from {workflow_dir}")

    elif args.command == "stats":
        show_stats(registry)

    elif args.command == "validate":
        is_valid = validate_workflows(registry)
        sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
