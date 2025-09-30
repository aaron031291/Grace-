#!/usr/bin/env python3
"""
validate_contracts.py
Validation script for Grace ML contracts & examples.

Features
- Validates example JSON docs against their JSON Schemas (draft-2020-12).
- Validates schema files against the meta-schema (where applicable).
- Resolves $ref across files (ml_schemas.json, ml_events.json).
- Clear per-test reporting, optional verbose traces, and proper exit codes.
- CLI flags: --contracts-dir, --fail-fast, --no-color, --verbose

Usage
  ./validate_contracts.py
  ./validate_contracts.py --contracts-dir ./grace-ml-contracts --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jsonschema import Draft202012Validator, Draft7Validator, validators, exceptions as js_exc, RefResolver

# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate Grace ML contract schemas and example payloads.")
    p.add_argument("--contracts-dir", type=str, default=str(Path(__file__).parent),
                   help="Directory containing ml_schemas.json, ml_events.json, ml_api.json, examples/")
    p.add_argument("--fail-fast", action="store_true", help="Stop on first failure.")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose error traces.")
    return p.parse_args()

# ---------- Colors ----------

class C:
    def __init__(self, enabled: bool):
        self.en = enabled
    def _c(self, code: str, s: str) -> str: return f"\033[{code}m{s}\033[0m" if self.en else s
    def ok(self, s: str) -> str:   return self._c("32;1", s)   # green bold
    def warn(self, s: str) -> str: return self._c("33;1", s)   # yellow bold
    def err(self, s: str) -> str:  return self._c("31;1", s)   # red bold
    def dim(self, s: str) -> str:  return self._c("90", s)     # gray

# ---------- IO helpers ----------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

# ---------- Schema validation helpers ----------

# Allow both 2020-12 and draft-07 (some files may still be 07)
META_VALIDATORS = {
    "https://json-schema.org/draft/2020-12/schema": Draft202012Validator,
    "http://json-schema.org/draft-07/schema#": Draft7Validator,
}

def validator_for_schema(schema: Dict[str, Any]) -> validators.Validator:
    # Choose by $schema, fallback to 2020-12
    meta_uri = schema.get("$schema", "https://json-schema.org/draft/2020-12/schema")
    V = META_VALIDATORS.get(meta_uri, Draft202012Validator)
    return V

def validate_schema_is_valid(schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    V = validator_for_schema(schema)
    try:
        V.check_schema(schema)
        return True, None
    except js_exc.SchemaError as e:
        return False, str(e)

# JSON Pointer resolver
def resolve_pointer(doc: Dict[str, Any], pointer: str) -> Any:
    """
    Resolve a local JSON Pointer like "#/AdaptationPlan" or "#/$defs/Experience".
    """
    if not pointer.startswith("#/"):
        raise ValueError(f"Unsupported pointer (must start with '#/'): {pointer}")
    parts = [p.replace("~1", "/").replace("~0", "~") for p in pointer[2:].split("/")]
    cur: Any = doc
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            raise KeyError(f"Pointer segment '{p}' not found")
    return cur

# ---------- Test plan ----------

@dataclass
class ExampleTest:
    filename: str
    schema_file: str
    pointer: str  # JSON Pointer to the definition inside schema_file

def build_tests() -> List[ExampleTest]:
    # Maps your examples to the canonical definitions in ml_schemas.json
    return [
        ExampleTest("adaptation_plan.json",     "ml_schemas.json", "#/AdaptationPlan"),
        ExampleTest("experience.json",          "ml_schemas.json", "#/Experience"),
        ExampleTest("insight.json",             "ml_schemas.json", "#/Insight"),
        ExampleTest("specialist_report.json",   "ml_schemas.json", "#/SpecialistReport"),
        ExampleTest("governance_snapshot.json", "ml_schemas.json", "#/GovernanceSnapshot"),
    ]

# ---------- Validation core ----------

def validate_example(
    contracts_dir: Path,
    example: ExampleTest,
    store: Dict[str, Dict[str, Any]],
    colors: C,
    verbose: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Validate a single example against its schema fragment.
    Returns (ok, messages).
    """
    ex_path = contracts_dir / "examples" / example.filename
    schema_path = contracts_dir / example.schema_file

    try:
        instance = load_json(ex_path)
    except Exception as e:
        return False, [colors.err(f"Unable to load {example.filename}: {e}")]

    try:
        schema_doc = load_json(schema_path)
    except Exception as e:
        return False, [colors.err(f"Unable to load schema {example.schema_file}: {e}")]

    try:
        fragment = resolve_pointer(schema_doc, example.pointer)
    except Exception as e:
        return False, [colors.err(f"Schema pointer {example.pointer} not found in {example.schema_file}: {e}")]

    # Build validator with a resolver so $ref across files can be followed.
    # We set a base_uri for relative refs to work (file://…).
    base_uri = schema_path.resolve().as_uri()
    resolver = RefResolver(base_uri=base_uri, referrer=schema_doc, store=store)

    V = validator_for_schema(schema_doc)
    validator = V(fragment, resolver=resolver)

    errors = sorted(validator.iter_errors(instance), key=lambda e: e.path)
    if not errors:
        msgs = [colors.ok(f"✅ {example.filename} ✓")]
        return True, msgs

    msgs = [colors.err(f"❌ {example.filename} failed validation: {len(errors)} error(s)")]
    if verbose:
        for err in errors:
            loc = "/".join([str(p) for p in err.path]) or "(root)"
            msgs.append(colors.dim(f"  at {loc}: {err.message}"))
            if err.context:
                for sub in err.context:
                    subloc = "/".join([str(p) for p in sub.path]) or "(root)"
                    msgs.append(colors.dim(f"    ↳ {subloc}: {sub.message}"))
    else:
        # just first error
        err = errors[0]
        loc = "/".join([str(p) for p in err.path]) or "(root)"
        msgs.append(colors.dim(f"  at {loc}: {err.message} (use --verbose for full trace)"))
    return False, msgs

# ---------- Main ----------

def main() -> int:
    args = parse_args()
    colors = C(enabled=not args.no_color)

    print(colors.dim("🧪 Grace ML Contract Validation Tests"))
    contracts_dir = Path(args.contracts_dir).resolve()

    # Load schemas into a store for $ref resolving
    store: Dict[str, Dict[str, Any]] = {}
    # Known schema files (add more here if needed)
    known = ["ml_schemas.json", "ml_events.json", "ml_api.json"]
    for name in known:
        p = contracts_dir / name
        try:
            doc = load_json(p)
            # Register by absolute file URI and by $id (if present)
            store[p.resolve().as_uri()] = doc
            if "$id" in doc:
                store[doc["$id"]] = doc
        except Exception as e:
            print(colors.err(f"❌ {name} is invalid JSON: {e}"))
            # keep going; we'll count as failure later
            store[p.resolve().as_uri()] = {}  # stub to keep resolver sane

    passed, failed = 0, 0

    # Validate schema files themselves (where they are JSON Schema; ml_api.json is more of a spec)
    for name in ["ml_schemas.json", "ml_events.json"]:
        doc = store.get((contracts_dir / name).resolve().as_uri())
        ok, err = (False, "missing") if not doc else validate_schema_is_valid(doc)
        if ok:
            print(colors.ok(f"✅ {name} schema is valid"))
            passed += 1
        else:
            print(colors.err(f"❌ {name} schema invalid: {err}"))
            failed += 1
            if args.fail_fast:
                print(colors.err("⛔ Fail-fast enabled — stopping early."))
                return 1

    # ml_api.json isn’t a pure JSON Schema doc; only check it parses
    try:
        _ = load_json(contracts_dir / "ml_api.json")
        print(colors.ok("✅ ml_api.json is valid JSON (spec parsed)"))
        passed += 1
    except Exception as e:
        print(colors.err(f"❌ ml_api.json is invalid JSON: {e}"))
        failed += 1
        if args.fail_fast:
            print(colors.err("⛔ Fail-fast enabled — stopping early."))
            return 1

    # Validate examples
    tests = build_tests()
    for t in tests:
        ok, msgs = validate_example(contracts_dir, t, store, colors, verbose=args.verbose)
        for m in msgs: print(m)
        if ok:
            passed += 1
        else:
            failed += 1
            if args.fail_fast:
                print(colors.err("⛔ Fail-fast enabled — stopping early."))
                return 1

    # Summary
    print()
    total = passed + failed
    print(colors.dim(f"📊 Validation results: {passed} passed, {failed} failed (of {total})"))
    if failed == 0:
        print(colors.ok("🎉 All validation tests passed!"))
        return 0
    else:
        print(colors.err("❌ Some validation tests failed!"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
