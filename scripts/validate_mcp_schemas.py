"""
Validate MCP schemas are properly defined
"""

import sys
from grace.mcp import MCPValidator


def main():
    """Validate all MCP schemas"""
    print("🔍 Validating MCP Schemas")
    print("=" * 60)
    
    validator = MCPValidator()
    
    # Check default schemas
    expected_schemas = ["heartbeat", "consensus_request", "consensus_response"]
    
    missing = []
    for schema_name in expected_schemas:
        if schema_name not in validator.schemas:
            missing.append(schema_name)
            print(f"❌ Missing schema: {schema_name}")
        else:
            schema = validator.schemas[schema_name]
            print(f"✅ Schema found: {schema_name} (v{schema.version})")
            print(f"   Required fields: {', '.join(schema.required_fields)}")
    
    print("\n" + "=" * 60)
    
    if missing:
        print(f"\n❌ Missing {len(missing)} required schemas")
        return 1
    else:
        print("\n✅ All MCP schemas valid!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
