#!/usr/bin/env python3
"""
Validation script to demonstrate Grace ML contract schemas.
Tests example JSON documents against their respective schemas.
"""

import json
import os
from pathlib import Path


def load_json_file(filepath):
    """Load and parse a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def validate_json_structure(data, required_fields):
    """Basic validation of JSON structure against required fields."""
    missing_fields = []
    for field in required_fields:
        if field not in data:
            missing_fields.append(field)
    return missing_fields


def main():
    """Run validation tests on contract examples."""
    print("🧪 Grace ML Contract Validation Tests\n")
    
    contracts_dir = Path(__file__).parent
    examples_dir = contracts_dir / "examples"
    
    # Load schemas
    schemas = load_json_file(contracts_dir / "ml_schemas.json")
    
    # Test cases: (filename, schema_name, required_fields)
    test_cases = [
        ("adaptation_plan.json", "AdaptationPlan", ["plan_id", "actions", "expected_effect", "risk_controls", "created_at", "version"]),
        ("experience.json", "Experience", ["experience_id", "source", "task", "context", "signals", "timestamp"]),
        ("insight.json", "Insight", ["insight_id", "type", "scope", "evidence", "confidence", "timestamp"]),
        ("specialist_report.json", "SpecialistReport", ["report_id", "specialist", "task", "candidates", "timestamp", "version"]),
        ("governance_snapshot.json", "GovernanceSnapshot", ["snapshot_id", "instance_id", "version", "policies", "thresholds", "model_weights", "state_hash", "created_at"])
    ]
    
    passed = 0
    failed = 0
    
    for filename, schema_name, required_fields in test_cases:
        print(f"🔍 Testing {filename} against {schema_name} schema...")
        
        try:
            # Load example
            example_data = load_json_file(examples_dir / filename)
            
            # Basic structure validation
            missing_fields = validate_json_structure(example_data, required_fields)
            
            if missing_fields:
                print(f"❌ Missing required fields: {missing_fields}")
                failed += 1
            else:
                print(f"✅ {filename} structure valid")
                passed += 1
                
                # Additional specific validations
                if schema_name == "AdaptationPlan":
                    if "actions" in example_data and len(example_data["actions"]) > 0:
                        print(f"   - Contains {len(example_data['actions'])} actions")
                    
                elif schema_name == "Experience":
                    if example_data.get("source") in ["training", "inference", "governance", "ops"]:
                        print(f"   - Valid source: {example_data['source']}")
                    if example_data.get("task") in ["classification", "regression", "clustering", "dimred", "rl"]:
                        print(f"   - Valid task: {example_data['task']}")
                        
                elif schema_name == "SpecialistReport":
                    if "candidates" in example_data:
                        print(f"   - Contains {len(example_data['candidates'])} candidates")
                        
        except Exception as e:
            print(f"❌ Error validating {filename}: {e}")
            failed += 1
        
        print()
    
    # Test schema file validity
    print("🔍 Testing schema files...")
    schema_files = [
        "ml_schemas.json",
        "ml_events.json", 
        "ml_api.json"
    ]
    
    for schema_file in schema_files:
        try:
            load_json_file(contracts_dir / schema_file)
            print(f"✅ {schema_file} is valid JSON")
            passed += 1
        except Exception as e:
            print(f"❌ {schema_file} is invalid: {e}")
            failed += 1
    
    print(f"\n📊 Validation Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All validation tests passed!")
        return True
    else:
        print("❌ Some validation tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)