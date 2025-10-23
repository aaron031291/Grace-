"""
Human-Readable Formatter for Memory and Logs
Makes all system data understandable to humans
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class HumanReadableFormatter:
    """
    Formats technical data into human-readable narratives
    """
    
    def __init__(self):
        self.templates = self._load_templates()
        logger.info("HumanReadableFormatter initialized")
    
    def _load_templates(self) -> Dict[str, str]:
        """Load human-readable templates"""
        return {
            'memory': "Memory '{id}' created on {date} by {actor}:\n  Content: {summary}\n  Confidence: {confidence}%",
            'decision': "Decision made on {date}:\n  Action: {action}\n  Reason: {reason}\n  Confidence: {confidence}%",
            'log_entry': "📝 {date}: {actor} performed '{action}'\n  ✅ Constitutional Check: {compliant}\n  📊 Result: {result}",
            'health_check': "🏥 Health Check on {date}:\n  Status: {status}\n  Components: {component_count}\n  Issues: {issues}",
            'consensus': "🤝 Consensus reached on {date}:\n  Decision: {decision}\n  Support: {support}%\n  Participants: {participants}"
        }
    
    def format_memory(self, memory: Dict[str, Any]) -> str:
        """Format memory entry for human reading"""
        date = self._format_date(memory.get('created_at', datetime.now()))
        
        # Extract key information
        content = memory.get('content', {})
        summary = self._summarize_dict(content, max_length=100)
        
        return f"""
📚 Memory Entry: {memory.get('memory_id', 'unknown')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📅 Created: {date}
👤 Source: {memory.get('source_node', 'unknown')}
🏷️  Type: {memory.get('memory_type', 'unknown')}
💯 Confidence: {memory.get('confidence', 0):.1%}

📋 Content:
{self._format_content(content)}

🔗 Relationships: {len(memory.get('relationships', []))}
👁️  Access Count: {memory.get('access_count', 0)}
"""
    
    def format_log_entry(self, log: Dict[str, Any]) -> str:
        """Format log entry for human reading"""
        date = self._format_date(log.get('timestamp', datetime.now()))
        compliant = "✅ Yes" if log.get('constitutional_check') else "⚠️ No"
        
        return f"""
📝 Audit Log Entry
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🕐 Time: {date}
👤 Actor: {log.get('actor', 'unknown')}
🎬 Action: {log.get('action', 'unknown')}
⚖️  Constitutional: {compliant}
🔒 Hash: {log.get('current_hash', '')[:16]}...

📊 Details:
{self._format_content(log.get('data', {}))}
"""
    
    def format_decision(self, decision: Dict[str, Any]) -> str:
        """Format decision for human reading"""
        date = self._format_date(decision.get('timestamp', datetime.now()))
        
        return f"""
🎯 Decision Made
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📅 Date: {date}
🎨 Source: {decision.get('source', 'unknown')}
💯 Confidence: {decision.get('confidence', 0):.1%}

✨ Decision: {decision.get('decision', 'N/A')}

💭 Rationale:
{decision.get('rationale', 'No rationale provided')}

🔍 Context:
{self._format_content(decision.get('metadata', {}))}
"""
    
    def format_health_status(self, health: Dict[str, Any]) -> str:
        """Format health status for human reading"""
        status_emoji = {
            'healthy': '✅',
            'degraded': '⚠️',
            'critical': '🔴',
            'offline': '💀'
        }
        
        status = health.get('status', 'unknown')
        emoji = status_emoji.get(status, '❓')
        
        return f"""
🏥 System Health Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{emoji} Overall Status: {status.upper()}
📊 Components Monitored: {health.get('components', 0)}

🔧 Component Health:
{self._format_component_health(health.get('component_statuses', {}))}

📈 Metrics:
{self._format_content(health.get('metrics', {}))}
"""
    
    def format_consensus(self, consensus: Dict[str, Any]) -> str:
        """Format consensus result for human reading"""
        return f"""
🤝 Consensus Result
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✨ Decision: {consensus.get('decision', 'N/A')}
📊 Support: {consensus.get('consensus_strength', 0):.1%}
👥 Participants: {len(consensus.get('participating_specialists', []))}

🗳️  Votes:
  For: {consensus.get('votes_for', 0)}
  Against: {consensus.get('votes_against', 0)}

💭 Key Rationales:
{self._format_rationales(consensus.get('rationales', []))}
"""
    
    def format_timeline(self, events: List[Dict[str, Any]]) -> str:
        """Format event timeline for human reading"""
        lines = ["⏱️  Timeline of Events", "━" * 50, ""]
        
        for event in sorted(events, key=lambda e: e.get('timestamp', datetime.now())):
            date = self._format_date(event.get('timestamp', datetime.now()))
            lines.append(f"• {date}: {event.get('description', 'Unknown event')}")
        
        return "\n".join(lines)
    
    def _format_date(self, date) -> str:
        """Format date for human reading"""
        if isinstance(date, str):
            try:
                date = datetime.fromisoformat(date.replace('Z', '+00:00'))
            except:
                return date
        
        if isinstance(date, datetime):
            return date.strftime("%B %d, %Y at %I:%M %p")
        
        return str(date)
    
    def _format_content(self, content: Dict[str, Any], indent: int = 0) -> str:
        """Format dictionary content for human reading"""
        lines = []
        prefix = "  " * indent
        
        for key, value in content.items():
            key_formatted = key.replace('_', ' ').title()
            
            if isinstance(value, dict):
                lines.append(f"{prefix}📁 {key_formatted}:")
                lines.append(self._format_content(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}📋 {key_formatted}: [{len(value)} items]")
            elif isinstance(value, bool):
                lines.append(f"{prefix}{'✅' if value else '❌'} {key_formatted}: {value}")
            elif isinstance(value, (int, float)):
                if 0 <= value <= 1:
                    lines.append(f"{prefix}📊 {key_formatted}: {value:.1%}")
                else:
                    lines.append(f"{prefix}🔢 {key_formatted}: {value}")
            else:
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                lines.append(f"{prefix}💬 {key_formatted}: {value_str}")
        
        return "\n".join(lines)
    
    def _summarize_dict(self, data: Dict[str, Any], max_length: int = 100) -> str:
        """Create brief summary of dictionary"""
        items = []
        for key, value in list(data.items())[:3]:
            items.append(f"{key}: {str(value)[:30]}")
        
        summary = ", ".join(items)
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        
        return summary
    
    def _format_component_health(self, statuses: Dict[str, str]) -> str:
        """Format component health statuses"""
        lines = []
        status_emoji = {
            'healthy': '✅',
            'degraded': '⚠️',
            'critical': '🔴',
            'offline': '💀'
        }
        
        for component, status in statuses.items():
            emoji = status_emoji.get(status, '❓')
            lines.append(f"  {emoji} {component.replace('_', ' ').title()}: {status}")
        
        return "\n".join(lines) if lines else "  No component data"
    
    def _format_rationales(self, rationales: List[str]) -> str:
        """Format list of rationales"""
        if not rationales:
            return "  No rationales provided"
        
        lines = []
        for i, rationale in enumerate(rationales[:5], 1):
            lines.append(f"  {i}. {rationale}")
        
        if len(rationales) > 5:
            lines.append(f"  ... and {len(rationales) - 5} more")
        
        return "\n".join(lines)
    
    def export_narrative_report(
        self,
        data: Dict[str, Any],
        report_type: str = "comprehensive"
    ) -> str:
        """Export comprehensive narrative report"""
        report = []
        
        report.append("=" * 60)
        report.append("GRACE AI SYSTEM - HUMAN-READABLE REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {self._format_date(datetime.now())}")
        report.append("")
        
        if 'memories' in data:
            report.append("📚 MEMORY SUMMARY")
            report.append("-" * 60)
            for memory in data['memories'][:5]:
                report.append(self.format_memory(memory))
                report.append("")
        
        if 'logs' in data:
            report.append("📝 AUDIT LOG SUMMARY")
            report.append("-" * 60)
            for log in data['logs'][:5]:
                report.append(self.format_log_entry(log))
                report.append("")
        
        if 'decisions' in data:
            report.append("🎯 DECISION SUMMARY")
            report.append("-" * 60)
            for decision in data['decisions'][:5]:
                report.append(self.format_decision(decision))
                report.append("")
        
        if 'health' in data:
            report.append("🏥 HEALTH SUMMARY")
            report.append("-" * 60)
            report.append(self.format_health_status(data['health']))
            report.append("")
        
        report.append("=" * 60)
        report.append("END OF REPORT")
        report.append("=" * 60)
        
        return "\n".join(report)
