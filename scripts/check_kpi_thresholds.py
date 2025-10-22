"""
Check KPI thresholds for CI/CD gating
"""

import sys
import json


def main(report_file: str):
    """Check KPI thresholds"""
    print("🔍 Checking KPI Thresholds")
    print("=" * 60)
    
    with open(report_file, 'r') as f:
        report = json.load(f)
    
    health = report.get("overall_health")
    health_percentage = report.get("health_percentage", 0)
    failed_kpis = report.get("failed_kpis", [])
    
    print(f"Overall Health: {health} ({health_percentage:.1f}%)")
    print(f"Failed KPIs: {len(failed_kpis)}")
    
    if failed_kpis:
        print("\nFailed KPIs:")
        for kpi in failed_kpis:
            print(f"  ❌ {kpi['name']}: {kpi['value']:.2f} (target: {kpi['target']:.2f})")
    
    print("\n" + "=" * 60)
    
    # Determine if build should fail
    if health_percentage < 70:
        print("\n❌ KPI health below 70% - FAILING BUILD")
        return 1
    elif health_percentage < 90:
        print("\n⚠️  KPI health below 90% - WARNING")
        return 0
    else:
        print("\n✅ All KPIs healthy!")
        return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_kpi_thresholds.py <report.json>")
        sys.exit(1)
    
    sys.exit(main(sys.argv[1]))
