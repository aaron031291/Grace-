"""Internationalization support for Interface Kernel."""
import json
import os
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class I18nManager:
    """Manages localization and message catalogs."""
    
    def __init__(self, locales_dir: Optional[str] = None):
        self.locales_dir = locales_dir or os.path.join(os.path.dirname(__file__), "locales")
        self.catalogs: Dict[str, Dict] = {}
        self.default_locale = "en-GB"
        self.current_locale = self.default_locale
        
        # Load available locales
        self._load_locales()
    
    def _load_locales(self):
        """Load all available locale files."""
        if not os.path.exists(self.locales_dir):
            logger.warning(f"Locales directory not found: {self.locales_dir}")
            return
        
        for filename in os.listdir(self.locales_dir):
            if filename.endswith('.json'):
                locale_code = filename[:-5]  # Remove .json extension
                locale_path = os.path.join(self.locales_dir, filename)
                
                try:
                    with open(locale_path, 'r', encoding='utf-8') as f:
                        catalog = json.load(f)
                        self.catalogs[locale_code] = catalog
                        logger.info(f"Loaded locale: {locale_code}")
                        
                except Exception as e:
                    logger.error(f"Failed to load locale {locale_code}: {e}")
    
    def set_locale(self, locale_code: str) -> bool:
        """Set current locale."""
        if locale_code in self.catalogs:
            self.current_locale = locale_code
            return True
        
        logger.warning(f"Locale {locale_code} not available, using {self.default_locale}")
        return False
    
    def get_message(self, key: str, locale: Optional[str] = None, **kwargs) -> str:
        """Get localized message by key."""
        locale = locale or self.current_locale
        
        # Try current locale first
        if locale in self.catalogs:
            message = self._get_nested_value(self.catalogs[locale], key)
            if message:
                return self._format_message(message, **kwargs)
        
        # Fall back to default locale
        if self.default_locale in self.catalogs and locale != self.default_locale:
            message = self._get_nested_value(self.catalogs[self.default_locale], key)
            if message:
                return self._format_message(message, **kwargs)
        
        # Return key if no translation found
        logger.warning(f"No translation found for key: {key}")
        return key
    
    def _get_nested_value(self, catalog: Dict, key: str) -> Optional[str]:
        """Get value from nested dictionary using dot notation."""
        keys = key.split('.')
        value = catalog
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        
        return value if isinstance(value, str) else None
    
    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with provided arguments."""
        try:
            return message.format(**kwargs)
        except (KeyError, ValueError) as e:
            logger.warning(f"Message formatting failed: {e}")
            return message
    
    def format_date(self, date: datetime, locale: Optional[str] = None) -> str:
        """Format date according to locale settings."""
        locale = locale or self.current_locale
        
        if locale in self.catalogs:
            format_str = self.catalogs[locale].get("formats", {}).get("date", "dd/MM/yyyy")
            
            # Convert format string from locale format to Python format
            py_format = format_str.replace("dd", "%d").replace("MM", "%m").replace("yyyy", "%Y")
            return date.strftime(py_format)
        
        return date.strftime("%d/%m/%Y")
    
    def format_time(self, time: datetime, locale: Optional[str] = None) -> str:
        """Format time according to locale settings."""
        locale = locale or self.current_locale
        
        if locale in self.catalogs:
            format_str = self.catalogs[locale].get("formats", {}).get("time", "HH:mm:ss")
            
            # Convert format string from locale format to Python format  
            py_format = format_str.replace("HH", "%H").replace("mm", "%M").replace("ss", "%S")
            return time.strftime(py_format)
        
        return time.strftime("%H:%M:%S")
    
    def format_datetime(self, dt: datetime, locale: Optional[str] = None) -> str:
        """Format datetime according to locale settings."""
        locale = locale or self.current_locale
        
        if locale in self.catalogs:
            format_str = self.catalogs[locale].get("formats", {}).get("datetime", "dd/MM/yyyy HH:mm")
            
            # Convert format string  
            py_format = (format_str
                        .replace("dd", "%d")
                        .replace("MM", "%m") 
                        .replace("yyyy", "%Y")
                        .replace("HH", "%H")
                        .replace("mm", "%M"))
            return dt.strftime(py_format)
        
        return dt.strftime("%d/%m/%Y %H:%M")
    
    def format_number(self, number: float, locale: Optional[str] = None) -> str:
        """Format number according to locale settings."""
        locale = locale or self.current_locale
        
        if locale in self.catalogs:
            formats = self.catalogs[locale].get("formats", {})
            decimal_sep = formats.get("decimal_separator", ".")
            thousand_sep = formats.get("thousand_separator", ",")
            
            # Simple number formatting
            if number >= 1000:
                return f"{number:,.2f}".replace(",", "TEMP").replace(".", decimal_sep).replace("TEMP", thousand_sep)
            else:
                return f"{number:.2f}".replace(".", decimal_sep)
        
        return f"{number:,.2f}"
    
    def get_available_locales(self) -> List[str]:
        """Get list of available locale codes."""
        return list(self.catalogs.keys())
    
    def get_locale_info(self, locale: str) -> Dict:
        """Get metadata about a locale."""
        if locale in self.catalogs:
            return self.catalogs[locale].get("meta", {})
        return {}
    
    def validate_locale_catalog(self, locale: str) -> Dict:
        """Validate locale catalog completeness."""
        if locale not in self.catalogs:
            return {"valid": False, "error": "Locale not found"}
        
        required_sections = ["interface", "formats"]
        missing_sections = []
        
        for section in required_sections:
            if section not in self.catalogs[locale]:
                missing_sections.append(section)
        
        return {
            "valid": len(missing_sections) == 0,
            "missing_sections": missing_sections,
            "total_keys": self._count_keys(self.catalogs[locale])
        }
    
    def _count_keys(self, catalog: Dict, prefix: str = "") -> int:
        """Count total translation keys in catalog."""
        count = 0
        for key, value in catalog.items():
            if isinstance(value, dict):
                count += self._count_keys(value, f"{prefix}.{key}" if prefix else key)
            elif isinstance(value, str):
                count += 1
        return count


# Global i18n manager instance
i18n_manager = I18nManager()

# Convenience functions
def _(key: str, **kwargs) -> str:
    """Get localized message (shorthand)."""
    return i18n_manager.get_message(key, **kwargs)

def set_locale(locale: str) -> bool:
    """Set current locale (shorthand)."""
    return i18n_manager.set_locale(locale)