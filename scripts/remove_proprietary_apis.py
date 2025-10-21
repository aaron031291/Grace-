"""
Remove all OpenAI and Anthropic API dependencies from Grace system
"""

import sys
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).parent.parent))


def remove_from_file(file_path: Path) -> int:
    """Remove OpenAI and Anthropic references from a file"""
    try:
        content = file_path.read_text()
        original = content
        
        # Remove OpenAI imports
        content = re.sub(r'import openai.*\n', '', content)
        content = re.sub(r'from openai.*\n', '', content)
        
        # Remove Anthropic imports
        content = re.sub(r'import anthropic.*\n', '', content)
        content = re.sub(r'from anthropic.*\n', '', content)
        
        # Remove OpenAI API key references
        content = re.sub(r'openai\.api_key.*\n', '', content)
        content = re.sub(r'OPENAI_API_KEY.*\n', '', content, flags=re.IGNORECASE)
        
        # Remove Anthropic API key references
        content = re.sub(r'ANTHROPIC_API_KEY.*\n', '', content, flags=re.IGNORECASE)
        
        # Comment out OpenAI-specific code blocks
        content = re.sub(
            r'(\s+)# OpenAI.*?\n(.*?)(class|def|\n\n)',
            r'\1# [REMOVED] OpenAI integration\n\3',
            content,
            flags=re.DOTALL
        )
        
        if content != original:
            file_path.write_text(content)
            return 1
        
        return 0
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0


def main():
    """Remove OpenAI and Anthropic from entire codebase"""
    print("\n" + "=" * 80)
    print("Removing OpenAI and Anthropic Dependencies")
    print("=" * 80)
    
    # Files to update
    files_to_check = [
        Path("requirements.txt"),
        Path("grace/config/settings.py"),
        Path("grace/embeddings/providers.py"),
        Path("grace/embeddings/service.py"),
        Path(".env.example"),
        Path("config/.env.template"),
    ]
    
    # Add all Python files in grace directory
    grace_dir = Path("grace")
    files_to_check.extend(grace_dir.rglob("*.py"))
    
    modified = 0
    
    for file_path in files_to_check:
        if file_path.exists():
            if remove_from_file(file_path):
                print(f"  ✓ Modified: {file_path}")
                modified += 1
    
    print(f"\n✅ Modified {modified} files")
    
    # Update requirements.txt specifically
    update_requirements()
    
    # Update settings.py specifically
    update_settings()
    
    # Update embeddings specifically
    update_embeddings()
    
    print("\n" + "=" * 80)
    print("Removal Complete!")
    print("=" * 80)
    
    print("\nChanges made:")
    print("  1. Removed OpenAI and Anthropic from requirements.txt")
    print("  2. Removed API key configurations")
    print("  3. Updated embedding providers to use only open-source alternatives")
    print("  4. Removed proprietary API calls")
    
    print("\nRemaining embedding options:")
    print("  - HuggingFace (sentence-transformers)")
    print("  - Local models")
    print("  - Custom embeddings")


def update_requirements():
    """Update requirements.txt to remove proprietary APIs"""
    req_file = Path("requirements.txt")
    
    if not req_file.exists():
        return
    
    lines = req_file.read_text().split('\n')
    new_lines = []
    
    for line in lines:
        # Skip OpenAI and Anthropic lines
        if 'openai' in line.lower() or 'anthropic' in line.lower():
            new_lines.append(f"# [REMOVED] {line}")
        else:
            new_lines.append(line)
    
    req_file.write_text('\n'.join(new_lines))
    print(f"  ✓ Updated: {req_file}")


def update_settings():
    """Update settings.py to remove OpenAI configuration"""
    settings_file = Path("grace/config/settings.py")
    
    if not settings_file.exists():
        return
    
    content = settings_file.read_text()
    
    # Update EmbeddingSettings to remove OpenAI
    new_content = content.replace(
        'provider: Literal["openai", "huggingface", "local"]',
        'provider: Literal["huggingface", "local"]'
    )
    
    # Remove OpenAI API key field
    new_content = re.sub(
        r'openai_api_key:.*?\n.*?description.*?\n',
        '# [REMOVED] OpenAI API key field\n',
        new_content,
        flags=re.DOTALL
    )
    
    # Remove OpenAI validation
    new_content = re.sub(
        r'@field_validator\(\'openai_api_key\'\).*?return v\n',
        '# [REMOVED] OpenAI validation\n',
        new_content,
        flags=re.DOTALL
    )
    
    settings_file.write_text(new_content)
    print(f"  ✓ Updated: {settings_file}")


def update_embeddings():
    """Update embedding providers to remove OpenAI"""
    providers_file = Path("grace/embeddings/providers.py")
    
    if not providers_file.exists():
        return
    
    content = providers_file.read_text()
    
    # Comment out OpenAIEmbeddings class
    new_content = re.sub(
        r'class OpenAIEmbeddings.*?(?=class|\Z)',
        '''# [REMOVED] OpenAI Embeddings
# class OpenAIEmbeddings(EmbeddingProvider):
#     """OpenAI embedding provider - REMOVED to eliminate proprietary dependencies"""
#     pass

''',
        content,
        flags=re.DOTALL
    )
    
    providers_file.write_text(new_content)
    print(f"  ✓ Updated: {providers_file}")
    
    # Update service.py
    service_file = Path("grace/embeddings/service.py")
    
    if service_file.exists():
        content = service_file.read_text()
        
        # Remove OpenAI provider initialization
        new_content = content.replace(
            'elif provider == "openai":',
            '# [REMOVED] OpenAI provider\n        # elif provider == "openai":'
        )
        
        new_content = re.sub(
            r'from \.providers import OpenAIEmbeddings',
            '# [REMOVED] from .providers import OpenAIEmbeddings',
            new_content
        )
        
        service_file.write_text(new_content)
        print(f"  ✓ Updated: {service_file}")


if __name__ == "__main__":
    sys.exit(main())
