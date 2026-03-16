"""VLM Response Parser.

Parses responses from various VLM models and extracts tags with confidence scores.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ParsedResponse:
    """Parsed VLM response with tags and metadata."""
    raw_response: str
    tags: List[str]
    confidence: Dict[str, float]
    parsing_method: str
    is_valid: bool
    error_message: Optional[str] = None


@dataclass
class TagCandidate:
    """A single tag candidate from VLM."""
    tag: str
    confidence: float
    position: int
    source: str  # Which parsing method found it


class VLMResponseParser:
    """Parser for VLM responses.
    
    Supports multiple parsing strategies:
    1. JSON format
    2. Comma-separated list
    3. Numbered list
    4. Pattern matching
    """
    
    # Tag extraction patterns
    PATTERNS = {
        "json_array": r'\[["\'](.*?)["\']\]',
        "comma_list": r'^([^\n,]+(?:,\s*[^\n,]+)*)',
        "numbered": r'^\d+[.)]\s*([^\n]+)',
        "bullet_points": r'^[-•*]\s*([^\n]+)',
        "tag_pattern": r'["\']([^"\']+)["\']',
    }
    
    # Chinese stop words to filter
    STOP_WORDS = {
        "的", "了", "是", "在", "有", "和", "與", "及", "或", "等",
        "圖片", "圖像", "圖中", "畫面", "可以看到", "有",
    }
    
    # Known tags from the tag library (for validation)
    KNOWN_TAGS = set()  # Will be populated from tag library
    
    def __init__(self, known_tags: Optional[set] = None):
        """Initialize parser.
        
        Args:
            known_tags: Set of known valid tags for validation
        """
        if known_tags:
            self.KNOWN_TAGS = known_tags
    
    def parse(
        self,
        response: str,
        expected_format: Optional[str] = None
    ) -> ParsedResponse:
        """Parse VLM response and extract tags.
        
        Args:
            response: Raw response from VLM
            expected_format: Expected format hint (json, list, etc.)
            
        Returns:
            ParsedResponse with tags and confidence scores
        """
        if not response or not response.strip():
            return ParsedResponse(
                raw_response="",
                tags=[],
                confidence={},
                parsing_method="empty",
                is_valid=False,
                error_message="Empty response",
            )
        
        # Try different parsing strategies
        strategies = [
            ("json", self._parse_json),
            ("comma_list", self._parse_comma_list),
            ("numbered", self._parse_numbered),
            ("bullet", self._parse_bullet),
            ("pattern", self._parse_pattern),
        ]
        
        for name, strategy in strategies:
            if expected_format and name != expected_format:
                continue
            
            try:
                result = strategy(response)
                if result.tags:
                    result.parsing_method = name
                    # Estimate confidence for each tag
                    result.confidence = self._estimate_confidence(response, result.tags)
                    result.is_valid = True
                    logger.debug(f"Parsed with {name}: {len(result.tags)} tags")
                    return result
            except Exception as e:
                logger.debug(f"Strategy {name} failed: {e}")
                continue
        
        # Fallback: extract any recognizable tags
        result = self._extract_any_tags(response)
        result.parsing_method = "fallback"
        result.is_valid = len(result.tags) > 0
        return result
    
    def _parse_json(self, response: str) -> ParsedResponse:
        """Parse JSON array format."""
        # Try to find JSON array
        patterns = [
            r'\["([^"]+)"(?:,\s*"([^"]+)")*\]',
            r"\['([^']+)'(?:,\s*'([^']+)')*\]",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                tags = []
                for group in match.groups():
                    if group:
                        tags.append(group.strip())
                if tags:
                    return ParsedResponse(
                        raw_response=response,
                        tags=tags,
                        confidence={},
                        parsing_method="json",
                        is_valid=True,
                    )
        
        raise ValueError("No JSON array found")
    
    def _parse_comma_list(self, response: str) -> ParsedResponse:
        """Parse comma-separated list."""
        # Clean response
        lines = response.strip().split('\n')
        first_line = lines[0].strip()
        
        # Remove common prefixes
        prefixes = [
            "Tags:",
            "標籤:",
            "Tags are:",
            "以下是標籤:",
            "Output:",
        ]
        for prefix in prefixes:
            if first_line.startswith(prefix):
                first_line = first_line[len(prefix):].strip()
        
        # Split by comma
        parts = re.split(r'[,，]', first_line)
        tags = [self._clean_tag(p) for p in parts if self._is_valid_tag(p)]
        
        return ParsedResponse(
            raw_response=response,
            tags=tags,
            confidence={},
            parsing_method="comma_list",
            is_valid=True,
        )
    
    def _parse_numbered(self, response: str) -> ParsedResponse:
        """Parse numbered list (1. tag, 2. tag, etc.)."""
        pattern = r'^\d+[.)]\s*([^\n]+)'
        lines = response.strip().split('\n')
        
        tags = []
        for i, line in enumerate(lines):
            match = re.match(pattern, line.strip())
            if match:
                tag = self._clean_tag(match.group(1))
                if self._is_valid_tag(tag):
                    tags.append(tag)
        
        if not tags:
            raise ValueError("No numbered list found")
        
        return ParsedResponse(
            raw_response=response,
            tags=tags,
            confidence={},
            parsing_method="numbered",
            is_valid=True,
        )
    
    def _parse_bullet(self, response: str) -> ParsedResponse:
        """Parse bullet point list (- tag, * tag, • tag)."""
        pattern = r'^[-•*]\s*([^\n]+)'
        lines = response.strip().split('\n')
        
        tags = []
        for line in lines:
            match = re.match(pattern, line.strip())
            if match:
                tag = self._clean_tag(match.group(1))
                if self._is_valid_tag(tag):
                    tags.append(tag)
        
        if not tags:
            raise ValueError("No bullet points found")
        
        return ParsedResponse(
            raw_response=response,
            tags=tags,
            confidence={},
            parsing_method="bullet",
            is_valid=True,
        )
    
    def _parse_pattern(self, response: str) -> ParsedResponse:
        """Parse using general patterns."""
        # Look for quoted strings
        pattern = r'["\']([^"\']+)["\']'
        matches = re.findall(pattern, response)
        
        tags = [self._clean_tag(m) for m in matches if self._is_valid_tag(m)]
        
        if not tags:
            raise ValueError("No quoted strings found")
        
        return ParsedResponse(
            raw_response=response,
            tags=tags,
            confidence={},
            parsing_method="pattern",
            is_valid=True,
        )
    
    def _extract_any_tags(self, response: str) -> ParsedResponse:
        """Fallback: extract anything that looks like a tag."""
        # Split by common delimiters
        parts = re.split(r'[,，\n]', response)
        
        tags = []
        for part in parts:
            # Clean and validate
            cleaned = self._clean_tag(part)
            if cleaned and len(cleaned) >= 2:
                tags.append(cleaned)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)
        
        return ParsedResponse(
            raw_response=response,
            tags=unique_tags,
            confidence={},
            parsing_method="fallback",
            is_valid=len(unique_tags) > 0,
        )
    
    def _clean_tag(self, text: str) -> str:
        """Clean and normalize a tag."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common prefixes/suffixes
        text = re.sub(r'^(tag|標籤)[:：]\s*', '', text, flags=re.IGNORECASE)
        
        # Remove trailing punctuation
        text = re.sub(r'[.,，。:：]+$', '', text)
        text = re.sub(r'^[.,，。:：]+', '', text)
        
        # Remove quotes
        text = re.sub(r'^["\']|["\']$', '', text)
        
        return text.strip()
    
    def _is_valid_tag(self, text: str) -> bool:
        """Check if text is a valid tag."""
        if not text or len(text) < 2:
            return False
        
        # Check for stop words
        for stop in self.STOP_WORDS:
            if text == stop:
                return False
        
        # Check length
        if len(text) > 20:
            return False
        
        # Check for common non-tag patterns
        non_tag_patterns = [
            r'^以下是',
            r'^圖片中',
            r'^可以看到',
            r'^I see',
            r'^The image',
        ]
        for pattern in non_tag_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return False
        
        return True
    
    def _estimate_confidence(
        self,
        raw_response: str,
        tags: List[str]
    ) -> Dict[str, float]:
        """Estimate confidence for each tag based on position and context."""
        confidence = {}
        
        for i, tag in enumerate(tags):
            # Position-based confidence (earlier = higher)
            position_score = 1.0 - (i * 0.05)
            position_score = max(position_score, 0.5)
            
            # Length-based confidence (reasonable length = higher)
            length = len(tag)
            if 2 <= length <= 8:
                length_score = 1.0
            elif length > 8:
                length_score = 0.8
            else:
                length_score = 0.6
            
            # Check if tag appears multiple times
            count = raw_response.lower().count(tag.lower())
            repetition_score = min(1.0, 0.5 + count * 0.25)
            
            # Combined confidence
            confidence[tag] = round(
                position_score * 0.5 +
                length_score * 0.3 +
                repetition_score * 0.2,
                3
            )
        
        return confidence
    
    def validate_tags(self, tags: List[str]) -> Tuple[List[str], List[str]]:
        """Validate tags against known tag library.
        
        Args:
            tags: List of tags to validate
            
        Returns:
            (valid_tags, invalid_tags)
        """
        valid = []
        invalid = []
        
        for tag in tags:
            # Normalize
            normalized = tag.lower().strip()
            
            # Check against known tags
            if normalized in {t.lower() for t in self.KNOWN_TAGS}:
                # Find exact match
                for known in self.KNOWN_TAGS:
                    if known.lower() == normalized:
                        valid.append(known)
                        break
            else:
                # Check if it's close to any known tag
                matched = self._fuzzy_match(tag)
                if matched:
                    valid.append(matched)
                else:
                    invalid.append(tag)
        
        return valid, invalid
    
    def _fuzzy_match(self, tag: str, threshold: float = 0.8) -> Optional[str]:
        """Fuzzy match a tag against known tags."""
        from difflib import SequenceMatcher
        
        tag_lower = tag.lower()
        best_match = None
        best_ratio = 0
        
        for known in self.KNOWN_TAGS:
            ratio = SequenceMatcher(None, tag_lower, known.lower()).ratio()
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_match = known
        
        return best_match
    
    def set_known_tags(self, tags: set):
        """Set the known tags from tag library."""
        self.KNOWN_TAGS = tags


# Singleton instance
_parser: Optional[VLMResponseParser] = None


def get_vlm_parser() -> VLMResponseParser:
    """Get or create VLM parser singleton."""
    global _parser
    if _parser is None:
        _parser = VLMResponseParser()
    return _parser
