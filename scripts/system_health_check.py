import requests
import sys

SERVICES = {
    "api": "http://localhost:8080/health",
    "governance_kernel": "http://localhost:8080/governance/health",
    "memory": "http://localhost:8080/memory/health",
    "learning_loop": "http://localhost:8080/learning/health",
    "llm": "http://localhost:8080/llm/health",
}


def check_service(name, url):
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            print(f"{name}: ACTIVE")
        else:
            print(f"{name}: ERROR (status {resp.status_code})")
    except Exception as e:
        print(f"{name}: ERROR ({e})")


def main():
    print("Grace System Health Check:")
    for name, url in SERVICES.items():
        check_service(name, url)


if __name__ == "__main__":
    main()
