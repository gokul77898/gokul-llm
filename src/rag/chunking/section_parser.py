"""
Section Parser for Legal Documents

Phase R1: Legal Chunking & Indexing

Extracts and parses legal section references from text.
Supports various section formats used in Indian legal documents.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ParsedSection:
    """
    Parsed section information.
    
    Attributes:
        section: Main section number (e.g., "420")
        subsection: Subsection identifier (e.g., "(1)")
        clause: Clause identifier (e.g., "(a)")
        text: Full text of the section
        start_offset: Start position in original document
        end_offset: End position in original document
        section_type: Type of section (section, explanation, illustration, proviso)
    """
    section: str
    subsection: Optional[str]
    clause: Optional[str]
    text: str
    start_offset: int
    end_offset: int
    section_type: str = "section"


class SectionParser:
    """
    Parser for legal section references.
    
    Supports:
    - "Section 420"
    - "Section 420(1)"
    - "Section 420(1)(a)"
    - "Explanation"
    - "Illustration"
    - "Proviso"
    """
    
    # Pattern for section headers - matches "Section X." at start of line or after newline
    SECTION_HEADER_PATTERN = re.compile(
        r'(?:^|\n)\s*Section\s+(\d+[A-Z]?)\s*\.?\s*',
        re.IGNORECASE
    )
    
    # Pattern for subsection markers
    SUBSECTION_PATTERN = re.compile(
        r'^\s*\((\d+)\)\s*',
        re.MULTILINE
    )
    
    # Pattern for clause markers
    CLAUSE_PATTERN = re.compile(
        r'^\s*\(([a-z])\)\s*',
        re.MULTILINE
    )
    
    # Pattern for special sections
    SPECIAL_SECTION_PATTERNS = {
        'explanation': re.compile(r'^Explanation\.?\s*[-—:]?\s*', re.IGNORECASE | re.MULTILINE),
        'illustration': re.compile(r'^Illustration\.?\s*[-—:]?\s*', re.IGNORECASE | re.MULTILINE),
        'proviso': re.compile(r'^Proviso\.?\s*[-—:]?\s*', re.IGNORECASE | re.MULTILINE),
        'exception': re.compile(r'^Exception\.?\s*[-—:]?\s*', re.IGNORECASE | re.MULTILINE),
    }
    
    def parse_bare_act(self, text: str) -> List[ParsedSection]:
        """
        Parse a bare act into sections.
        
        Args:
            text: Full text of the bare act
            
        Returns:
            List of ParsedSection objects
        """
        sections: List[ParsedSection] = []
        
        # Find all section boundaries
        section_matches = list(self.SECTION_HEADER_PATTERN.finditer(text))
        
        if not section_matches:
            # No sections found - return entire text as single section
            return [ParsedSection(
                section="1",
                subsection=None,
                clause=None,
                text=text.strip(),
                start_offset=0,
                end_offset=len(text),
                section_type="section"
            )]
        
        # Process each section
        for i, match in enumerate(section_matches):
            section_num = match.group(1)
            start_offset = match.start()
            
            # End is either next section or end of text
            if i + 1 < len(section_matches):
                end_offset = section_matches[i + 1].start()
            else:
                end_offset = len(text)
            
            section_text = text[start_offset:end_offset].strip()
            
            # Parse subsections within this section
            subsections = self._parse_subsections(section_text, section_num, start_offset)
            
            if subsections:
                sections.extend(subsections)
            else:
                # No subsections - add as single section
                sections.append(ParsedSection(
                    section=section_num,
                    subsection=None,
                    clause=None,
                    text=section_text,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    section_type="section"
                ))
        
        return sections
    
    def _parse_subsections(
        self,
        section_text: str,
        section_num: str,
        base_offset: int
    ) -> List[ParsedSection]:
        """
        Parse subsections within a section.
        
        Args:
            section_text: Text of the section
            section_num: Section number
            base_offset: Offset of section start in original document
            
        Returns:
            List of ParsedSection for subsections (empty if none found)
        """
        subsections: List[ParsedSection] = []
        
        # Find subsection markers like (1), (2), etc.
        subsection_matches = list(self.SUBSECTION_PATTERN.finditer(section_text))
        
        # Also find special sections (Explanation, Illustration, etc.)
        special_matches = []
        for special_type, pattern in self.SPECIAL_SECTION_PATTERNS.items():
            for match in pattern.finditer(section_text):
                special_matches.append((match.start(), match.end(), special_type, match))
        
        # Sort special matches by position
        special_matches.sort(key=lambda x: x[0])
        
        # If we have subsections, parse them
        if len(subsection_matches) > 1:  # Need at least 2 to have boundaries
            for i, match in enumerate(subsection_matches):
                subsection_num = match.group(1)
                start = match.start()
                
                # Find end (next subsection, special section, or end)
                end = len(section_text)
                
                # Check next subsection
                if i + 1 < len(subsection_matches):
                    end = min(end, subsection_matches[i + 1].start())
                
                # Check special sections
                for sp_start, sp_end, sp_type, sp_match in special_matches:
                    if sp_start > start:
                        end = min(end, sp_start)
                        break
                
                subsection_text = section_text[start:end].strip()
                
                if subsection_text:
                    subsections.append(ParsedSection(
                        section=section_num,
                        subsection=f"({subsection_num})",
                        clause=None,
                        text=subsection_text,
                        start_offset=base_offset + start,
                        end_offset=base_offset + end,
                        section_type="subsection"
                    ))
        
        # Add special sections (Explanation, Illustration, etc.)
        for i, (sp_start, sp_end, sp_type, sp_match) in enumerate(special_matches):
            # Find end
            end = len(section_text)
            if i + 1 < len(special_matches):
                end = special_matches[i + 1][0]
            
            special_text = section_text[sp_start:end].strip()
            
            if special_text:
                subsections.append(ParsedSection(
                    section=section_num,
                    subsection=sp_type,
                    clause=None,
                    text=special_text,
                    start_offset=base_offset + sp_start,
                    end_offset=base_offset + end,
                    section_type=sp_type
                ))
        
        return subsections
    
    def parse_case_law(self, text: str) -> List[ParsedSection]:
        """
        Parse case law into paragraph-level chunks.
        
        Case law is chunked by:
        - Numbered paragraphs
        - Major headings (JUDGMENT, HELD, etc.)
        - Double newline boundaries
        
        Args:
            text: Full text of the case law
            
        Returns:
            List of ParsedSection objects
        """
        sections: List[ParsedSection] = []
        
        # Pattern for numbered paragraphs: "1.", "2.", etc.
        para_pattern = re.compile(r'^\s*(\d+)\.\s+', re.MULTILINE)
        
        # Pattern for major headings
        heading_pattern = re.compile(
            r'^(JUDGMENT|HELD|ORDER|FACTS|ISSUES?|RATIO|CONCLUSION|ARGUMENTS?)\s*:?\s*$',
            re.IGNORECASE | re.MULTILINE
        )
        
        # Find all paragraph markers
        para_matches = list(para_pattern.finditer(text))
        heading_matches = list(heading_pattern.finditer(text))
        
        # Combine and sort all markers
        all_markers: List[Tuple[int, str, str]] = []
        
        for match in para_matches:
            all_markers.append((match.start(), f"para_{match.group(1)}", "paragraph"))
        
        for match in heading_matches:
            all_markers.append((match.start(), match.group(1).lower(), "heading"))
        
        all_markers.sort(key=lambda x: x[0])
        
        if not all_markers:
            # No markers - split by double newlines
            paragraphs = text.split('\n\n')
            offset = 0
            for i, para in enumerate(paragraphs):
                para = para.strip()
                if para and len(para) > 20:  # Skip very short paragraphs
                    sections.append(ParsedSection(
                        section=str(i + 1),
                        subsection=None,
                        clause=None,
                        text=para,
                        start_offset=offset,
                        end_offset=offset + len(para),
                        section_type="paragraph"
                    ))
                offset += len(para) + 2  # +2 for \n\n
            return sections
        
        # Process markers
        for i, (start, marker, marker_type) in enumerate(all_markers):
            # Find end
            if i + 1 < len(all_markers):
                end = all_markers[i + 1][0]
            else:
                end = len(text)
            
            para_text = text[start:end].strip()
            
            if para_text and len(para_text) > 20:
                sections.append(ParsedSection(
                    section=marker,
                    subsection=None,
                    clause=None,
                    text=para_text,
                    start_offset=start,
                    end_offset=end,
                    section_type=marker_type
                ))
        
        return sections
    
    def extract_section_number(self, text: str) -> Optional[str]:
        """
        Extract section number from text.
        
        Args:
            text: Text that may contain a section reference
            
        Returns:
            Section number or None
        """
        match = self.SECTION_HEADER_PATTERN.search(text)
        if match:
            return match.group(1)
        return None
    
    def validate_section(self, section: str) -> bool:
        """
        Validate a section number format.
        
        Valid formats:
        - "420"
        - "420A"
        - "420(1)"
        - "420(1)(a)"
        
        Args:
            section: Section number to validate
            
        Returns:
            True if valid
        """
        pattern = r'^(\d+[A-Z]?(?:\([0-9a-z]+\))*)$'
        return bool(re.match(pattern, section, re.IGNORECASE))
