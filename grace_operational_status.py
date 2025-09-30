#!/usr/bin/env python3
"""
Grace System Operational Status - Final Assessment
=================================================

This script provides a definitive answer to: "Is the Grace system operational?"

Usage:
    python grace_operational_status.py

Author: Grace System Analysis Team
Version: 1.0.0
Date: September 2025
"""

import asyncio
import sys
from datetime import datetime
from grace_operational_analysis import GraceOperationalAnalyzer
from pathlib import Path


def print_operational_status_summary(analysis):
    """Print a clear, concise operational status summary."""
    
    print("\n" + "="*80)
    print("🏛️  GRACE SYSTEM OPERATIONAL STATUS - FINAL ASSESSMENT")
    print("="*80)
    
    print(f"\n📅 Assessment Date: {analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Main Question Answer
    print(f"\n❓ QUESTION: Is the Grace Governance System operational?")
    print("-" * 60)
    
    if analysis.overall_operational_status == "FULLY OPERATIONAL":
        print("✅ ANSWER: YES - The Grace system is FULLY OPERATIONAL")
        operational_verdict = "YES"
    elif analysis.overall_operational_status == "OPERATIONAL":
        print("✅ ANSWER: YES - The Grace system is OPERATIONAL with minor issues")
        operational_verdict = "YES"
    elif analysis.overall_operational_status == "PARTIALLY OPERATIONAL":
        print("⚠️  ANSWER: PARTIALLY - The Grace system has limited operational capability")
        operational_verdict = "PARTIALLY"
    else:
        print("❌ ANSWER: NO - The Grace system is NOT operational")
        operational_verdict = "NO"
    
    # Key Metrics
    print(f"\n📊 KEY OPERATIONAL METRICS")
    print("-" * 40)
    print(f"Overall Operational Score: {analysis.operational_score:.1%}")
    print(f"Components Working: {analysis.operational_components}/{analysis.total_components} ({analysis.operational_components/analysis.total_components:.1%})")
    print(f"Dependencies Satisfied: {'✅ YES' if analysis.dependencies_satisfied else '❌ NO'}")
    print(f"Communication System: {'✅ WORKING' if analysis.communication_functional else '❌ FAILED'}")
    print(f"Governance System: {'✅ WORKING' if analysis.governance_functional else '❌ FAILED'}")
    
    # Production Readiness
    print(f"\n🚀 PRODUCTION READINESS")
    print("-" * 30)
    if analysis.readiness_assessment.production_ready:
        print("✅ READY FOR PRODUCTION USE")
    else:
        print("❌ NOT READY FOR PRODUCTION")
        print("\n🔥 CRITICAL ISSUES TO RESOLVE:")
        for issue in analysis.readiness_assessment.critical_issues:
            print(f"   • {issue}")
        
        if analysis.readiness_assessment.immediate_actions:
            print("\n⚡ IMMEDIATE ACTIONS REQUIRED:")
            for action in analysis.readiness_assessment.immediate_actions:
                print(f"   • {action}")
    
    # Summary Recommendations
    print(f"\n💡 SUMMARY RECOMMENDATIONS")
    print("-" * 35)
    
    if operational_verdict == "YES":
        print("1. ✅ System is operational - continue normal operations")
        print("2. 📊 Monitor system health regularly")
        print("3. 🔧 Address any minor issues when convenient")
    elif operational_verdict == "PARTIALLY":
        print("1. ⚠️  System has limited functionality")
        print("2. 🔧 Fix failing components to restore full operation")
        print("3. 📊 Increase monitoring until issues are resolved")
        print("4. ⚡ Avoid production deployment until issues are fixed")
    else:
        print("1. ❌ System requires immediate attention")
        print("2. 🚨 Do not deploy to production")
        print("3. 🔧 Resolve critical issues before use")
        print("4. 📞 Consider escalating to system administrators")
    
    # System Health at a Glance
    print(f"\n🏥 SYSTEM HEALTH AT A GLANCE")
    print("-" * 35)
    
    working_components = [c for c in analysis.component_statuses if c.operational]
    failed_components = [c for c in analysis.component_statuses if not c.operational]
    
    if working_components:
        print(f"✅ WORKING COMPONENTS ({len(working_components)}):")
        for comp in working_components[:5]:  # Show first 5
            print(f"   • {comp.name}")
        if len(working_components) > 5:
            print(f"   • ... and {len(working_components) - 5} more")
    
    if failed_components:
        print(f"\n❌ FAILED COMPONENTS ({len(failed_components)}):")
        for comp in failed_components[:5]:  # Show first 5 with errors
            print(f"   • {comp.name}")
            if comp.error_details:
                error_summary = comp.error_details.split(':')[0] if ':' in comp.error_details else comp.error_details
                print(f"     └─ {error_summary}")
        if len(failed_components) > 5:
            print(f"   • ... and {len(failed_components) - 5} more")
    
    print(f"\n" + "="*80)
    
    # Final Verdict
    if operational_verdict == "YES":
        print("🎯 FINAL VERDICT: Grace system IS OPERATIONAL ✅")
    elif operational_verdict == "PARTIALLY":
        print("🎯 FINAL VERDICT: Grace system is PARTIALLY OPERATIONAL ⚠️")
    else:
        print("🎯 FINAL VERDICT: Grace system is NOT OPERATIONAL ❌")
    
    print("="*80)
    
    return operational_verdict


async def main():
    """Main function to assess Grace operational status."""
    
    print("🔍 Assessing Grace System Operational Status...")
    print("⏳ Running comprehensive analysis...")
    
    try:
        # Run the comprehensive operational analysis
        analyzer = GraceOperationalAnalyzer()
        analysis = await analyzer.run_operational_analysis(detailed=False)
        
        # Print the summary
        operational_verdict = print_operational_status_summary(analysis)
        
        # Exit with appropriate code
        if operational_verdict == "YES":
            print("\n✅ Exiting with success code (0)")
            sys.exit(0)
        elif operational_verdict == "PARTIALLY":
            print("\n⚠️  Exiting with warning code (1)")
            sys.exit(1)
        else:
            print("\n❌ Exiting with error code (2)")
            sys.exit(2)
            
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR during analysis: {e}")
        print("❌ Cannot determine operational status")
        print("🆘 System may be in an unknown state")
        sys.exit(3)


if __name__ == "__main__":
    asyncio.run(main())