#!/usr/bin/env python3
"""
Grace Test Quality Report Viewer

View test quality reports with KPI integration and component breakdowns.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import argparse


def load_latest_report():
    """Load the most recent quality report."""
    report_dir = Path("/workspaces/Grace-/test_reports")
    if not report_dir.exists():
        print("❌ No test reports found. Run pytest first.")
        sys.exit(1)
    
    report_files = sorted(report_dir.glob("quality_report_*.json"), reverse=True)
    if not report_files:
        print("❌ No quality reports found. Run pytest first.")
        sys.exit(1)
    
    with open(report_files[0]) as f:
        return json.load(f)


def print_summary(report):
    """Print quality summary."""
    summary = report['summary']
    
    print("\n" + "="*80)
    print("🎯 GRACE TEST QUALITY REPORT (90% Threshold Model)")
    print("="*80)
    print(f"\n📅 Report Time: {report['timestamp']}")
    
    # System-wide metrics
    system_pass_rate = summary['system_pass_rate'] * 100
    overall_quality = summary['overall_quality'] * 100
    
    print(f"\n📊 System-Wide Quality:")
    print(f"  Total Components:     {summary['total_components']}")
    print(f"  Passing Components:   {summary['passing_components']} (≥90% quality)")
    print(f"  System Pass Rate:     {system_pass_rate:.1f}% {'✅' if system_pass_rate >= 90 else '⚠️'}")
    print(f"  Overall Avg Quality:  {overall_quality:.1f}%")
    
    # Status breakdown
    print(f"\n📈 Component Status Breakdown:")
    breakdown = summary['breakdown']
    for status in ['excellent', 'passing', 'acceptable', 'degraded', 'critical']:
        count = breakdown.get(status, 0)
        emoji = {
            'excellent': '🌟',
            'passing': '✅',
            'acceptable': '⚡',
            'degraded': '⚠️',
            'critical': '🔴'
        }[status]
        print(f"  {emoji} {status.upper():12} {count:3} components")
    
    # Raw counts
    raw = report['raw_counts']
    print(f"\n📝 Raw Test Results:")
    print(f"  Total Tests:   {raw['total']}")
    print(f"  Passed:        {raw['passed']} ({raw['passed']/max(1, raw['total'])*100:.1f}%)")
    print(f"  Failed:        {raw['failed']} ({raw['failed']/max(1, raw['total'])*100:.1f}%)")
    print(f"  Skipped:       {raw['skipped']} ({raw['skipped']/max(1, raw['total'])*100:.1f}%)")


def print_components(report, show_all=False):
    """Print component details."""
    print(f"\n" + "="*80)
    print("📦 COMPONENT DETAILS")
    print("="*80)
    
    components = report['components']
    
    # Sort by score (lowest first if showing needing attention)
    sorted_components = sorted(
        components.items(),
        key=lambda x: x[1]['trust_adjusted_score'],
        reverse=show_all
    )
    
    for comp_id, comp in sorted_components:
        score = comp['trust_adjusted_score'] * 100
        status = comp['quality_status']
        
        emoji = {
            'excellent': '🌟',
            'passing': '✅',
            'acceptable': '⚡',
            'degraded': '⚠️',
            'critical': '🔴'
        }[status]
        
        print(f"\n{emoji} {comp_id}")
        print(f"  Quality Score:   {score:.1f}% ({status.upper()})")
        print(f"  Tests:           {comp['passed_tests']}/{comp['total_tests']} passed ({comp['pass_rate']*100:.1f}%)")
        
        if not comp['is_passing']:
            gap = (0.90 - comp['trust_adjusted_score']) * 100
            print(f"  Gap to Passing:  {gap:.1f}%")
        
        # Error breakdown
        errors = comp['error_severity_breakdown']
        error_count = sum(errors.values())
        if error_count > 0:
            print(f"  Errors:          {error_count} total")
            severity_names = {
                '1': 'Low',
                '2': 'Medium',
                '3': 'High',
                '4': 'Critical'
            }
            for sev, count in errors.items():
                if count > 0:
                    print(f"    - {severity_names.get(sev, sev)}: {count}")


def print_attention_needed(report):
    """Print components needing attention."""
    needing_attention = report['needing_attention']
    
    if not needing_attention:
        print(f"\n✅ All components passing! No attention needed.")
        return
    
    print(f"\n" + "="*80)
    print(f"⚠️  COMPONENTS NEEDING ATTENTION ({len(needing_attention)})")
    print("="*80)
    
    for i, comp in enumerate(needing_attention, 1):
        score = comp['score'] * 100
        gap = comp['gap_to_passing'] * 100
        priority = comp['priority']
        
        print(f"\n{i}. {comp['component_id']}")
        print(f"   Current Score:   {score:.1f}%")
        print(f"   Gap to Pass:     {gap:.1f}%")
        print(f"   Priority:        {priority:.0f}")
        print(f"   Status:          {comp['status'].upper()}")


def print_recommendations(report):
    """Print improvement recommendations."""
    needing_attention = report['needing_attention']
    
    if not needing_attention:
        return
    
    print(f"\n" + "="*80)
    print("💡 IMPROVEMENT RECOMMENDATIONS")
    print("="*80)
    
    # Focus on highest priority components
    top_components = sorted(needing_attention, key=lambda x: x['priority'], reverse=True)[:3]
    
    print(f"\nFocus on these components for maximum impact:\n")
    
    for i, comp in enumerate(top_components, 1):
        print(f"{i}. **{comp['component_id']}** (Priority: {comp['priority']:.0f})")
        gap = comp['gap_to_passing'] * 100
        
        if comp['status'] == 'critical':
            print(f"   🔴 CRITICAL: Escalate to AVN for immediate healing")
            print(f"   - Review recent code changes")
            print(f"   - Check system resources")
            print(f"   - Verify dependencies")
        elif comp['status'] == 'degraded':
            print(f"   ⚠️  DEGRADED: Trigger adaptive learning")
            print(f"   - Run diagnostic tests")
            print(f"   - Analyze failure patterns")
        else:
            print(f"   ⚡ Need {gap:.1f}% improvement to reach 90% threshold")
            print(f"   - Review and fix failing tests")
            print(f"   - Address high-severity errors first")


def main():
    parser = argparse.ArgumentParser(description='View Grace test quality reports')
    parser.add_argument('--all', action='store_true', help='Show all components')
    parser.add_argument('--components-only', action='store_true', help='Show only component details')
    parser.add_argument('--summary-only', action='store_true', help='Show only summary')
    args = parser.parse_args()
    
    report = load_latest_report()
    
    if not args.components_only:
        print_summary(report)
    
    if not args.summary_only:
        if args.all:
            print_components(report, show_all=True)
        else:
            print_attention_needed(report)
            print_recommendations(report)
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
