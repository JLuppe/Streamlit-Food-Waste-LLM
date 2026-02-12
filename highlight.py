from bs4 import BeautifulSoup

def get_text_positions_in_html(html_content, text_to_find):
    """Find where text appears in the plain text version of HTML"""
    soup = BeautifulSoup(html_content, 'html.parser')
    plain_text = soup.get_text()
    
    positions = []
    for text in text_to_find:
        text = text.strip()
        if not text:
            continue
        
        # Find all occurrences in plain text
        start = 0
        while True:
            pos = plain_text.find(text, start)
            if pos == -1:
                break
            positions.append((pos, pos + len(text), text))
            start = pos + 1
    
    return positions, plain_text

def highlight_by_text_position(html_content, chunks):
    """Highlight chunks in HTML based on plain text positions"""
    soup = BeautifulSoup(html_content, 'html.parser')
    plain_text = soup.get_text()
    
    # Sort chunks by length (longest first) to avoid partial matches
    sorted_chunks = sorted(chunks, key=len, reverse=True)
    
    # Track which positions we've already highlighted
    highlighted_ranges = []
    
    for chunk in sorted_chunks:
        chunk = chunk.strip()
        if not chunk or len(chunk) < 10:  # Skip very short chunks
            continue
        
        # Find chunk in plain text
        start_pos = plain_text.find(chunk)
        if start_pos == -1:
            continue
        
        end_pos = start_pos + len(chunk)
        
        # Check if this overlaps with already highlighted text
        overlaps = any(
            not (end_pos <= h_start or start_pos >= h_end)
            for h_start, h_end in highlighted_ranges
        )
        
        if overlaps:
            continue
        
        # Highlight this chunk
        highlighted_ranges.append((start_pos, end_pos))
        highlight_text_range(soup, start_pos, end_pos)
    
    return str(soup)

def highlight_text_range(soup, start_pos, end_pos):
    """Highlight text between start_pos and end_pos in plain text"""
    current_pos = 0
    
    for element in soup.find_all(string=True):
        if element.parent.name in ['script', 'style', 'mark']:
            continue
        
        text = str(element)
        text_start = current_pos
        text_end = current_pos + len(text)
        
        # Check if this text node overlaps with target range
        if text_end > start_pos and text_start < end_pos:
            # Calculate overlap within this text node
            highlight_start = max(0, start_pos - text_start)
            highlight_end = min(len(text), end_pos - text_start)
            
            # Create highlighted version
            before = text[:highlight_start]
            highlighted = text[highlight_start:highlight_end]
            after = text[highlight_end:]
            
            new_soup = BeautifulSoup(
                f'{before}<mark style="background-color: yellow;">{highlighted}</mark>{after}',
                'html.parser'
            )
            element.replace_with(new_soup)
        
        current_pos = text_end
