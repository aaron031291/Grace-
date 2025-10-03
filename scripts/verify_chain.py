"""
Golden Path Audit Chain Verifier
Checks the hash chain of audit logs for integrity.
"""
from grace.audit.immutable_logs import verify_audit_chain

def main():
    result = verify_audit_chain()
    if result["valid"]:
        print("Audit chain is valid.")
    else:
        print("Audit chain is INVALID! Details:", result["details"])

if __name__ == "__main__":
    main()
