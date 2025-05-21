import logging
import re
import yaml

import nltk
from nltk.tokenize import sent_tokenize
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

import config

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)

def is_mostly_html(text, threshold=0.5):
    """Checks if a text contains mostly HTML by calculating the percentage of removed content."""
    original_length = len(text)
    stripped_text = re.sub(r"<[^>]+>", "", text).strip()
    stripped_length = len(stripped_text)
    # If removing HTML leaves less than (threshold)% of the original content, it's mostly HTML
    return original_length > 0 and (stripped_length / original_length) < (1 - threshold)


def approximate_tokens(text):
    return len(config.tokenizer.tokenize(text, truncation=False))


def extract_yaml_info(text):
    """
    Extract YAML info from markdown text.
    Returns:
        tuple: (metadata_dict, content_without_YAML)
    """
    # Pattern to match frontmatter at the beginning of the file
    frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n'
    match = re.match(frontmatter_pattern, text, re.DOTALL)

    if match:
        frontmatter_text = match.group(1)
        content = text[match.end():]
        try:
            metadata = yaml.safe_load(frontmatter_text)
            # Ensure metadata is a dictionary
            if metadata is None or not isinstance(metadata, dict):
                metadata = {}
            return metadata, content
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML frontmatter: {e}")
            # Extract key-value pairs manually for malformed YAML
            metadata = {}
            for line in frontmatter_text.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
            return metadata, content
    return {}, text


def split_text_with_markdown_headers(text, chunk_size=config.chunk_size, chunk_overlap=config.overlap):
    """
    Split text based on markdown headers to maintain semantic structure.
    Extracts YAML yaml_data and adds section summaries to each chunk's metadata.

    Args:
        text: The text to split
        chunk_size: Maximum size of chunks in tokens
        chunk_overlap: Number of tokens of overlap between chunks

    Returns:
        List of text chunks and their section summaries
    """
    # Extract yaml_data first
    yaml_data, content = extract_yaml_info(text)

    # Define headers to split on - adjust based on your document structure
    headers_to_split_on = [
        ("#", "header1"),
        ("##", "header2"),
        ("###", "header3"),
        ("####", "header4"),
    ]

    # Create the markdown splitter
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False  # Keep headers in the chunks for context
    )

    try:
        header_documents = markdown_splitter.split_text(content)

        # If no headers were found, header_documents will be empty or just have one large chunk
        if not header_documents:
            logger.debug("No markdown headers found, using fallback chunking")
            chunks, chunk_metadata = split_text_with_overlap(content, chunk_size, chunk_overlap)
            # Add yaml_data to all chunk metadata
            for metadata in chunk_metadata:
                metadata["yaml_data"] = yaml_data
            return chunks, chunk_metadata

        # Create a secondary splitter for chunks that exceed token limit
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = []
        chunk_metadata = []
        for doc in header_documents:
            document_content = doc.page_content
            metadata = doc.metadata

            if is_mostly_html(document_content):
                chunks.append("")
                chunk_metadata.append({
                    "section_summary": "",
                    "yaml_data": yaml_data
                })
                continue

            section_hierarchy = {}
            header_prefix = ""
            for level in ["header1", "header2", "header3", "header4"]:
                if level in metadata and metadata[level]:
                    level_marker = "#" * int(level[-1])  # Convert header2 to ##, etc.
                    header_prefix += f"{level_marker} {metadata[level]}\n"
                    section_hierarchy[level] = metadata[level]

            section_summary = generate_section_summary(document_content, section_hierarchy)

            estimated_token_count = approximate_tokens(document_content)
            if estimated_token_count > chunk_size:
                logger.debug(f"Splitting section with estimated token count: {estimated_token_count}")
                sub_chunks = recursive_splitter.split_text(document_content)

                # Add the first chunk as is (it already has the content from the original)
                chunks.append(sub_chunks[0])
                chunk_metadata.append({
                    "section_summary": section_summary,
                    "yaml_data": yaml_data  # Add yaml_data to metadata
                })

                # For subsequent chunks, check if they already start with a header
                # to avoid header duplication
                for sub_chunk in sub_chunks[1:]:
                    has_header = False
                    for header_marker, _ in headers_to_split_on:
                        if sub_chunk.strip().startswith(header_marker + " "):
                            has_header = True
                            break

                    if not has_header and header_prefix:
                        chunks.append(f"{header_prefix}{sub_chunk}")
                    else:
                        chunks.append(sub_chunk)

                    # Add the same metadata for all sub-chunks from the same section
                    chunk_metadata.append({
                        "section_summary": section_summary,
                        "yaml_data": yaml_data
                    })
            else:
                chunks.append(document_content)
                chunk_metadata.append({
                    "section_summary": section_summary,
                    "yaml_data": yaml_data
                })

    except Exception as e:
        logger.exception(f"Error in markdown splitting: {e}")
        # Fall back to regular chunking
        chunks, chunk_metadata = split_text_with_overlap(content, chunk_size, chunk_overlap)
        # Add yaml_data to all chunk metadata
        for metadata in chunk_metadata:
            metadata["yaml_data"] = yaml_data
        return [], []

    return chunks, chunk_metadata


def generate_section_summary(section_text, section_hierarchy):
    """
    Generate a concise summary of a section based on its content and headers.

    Args:
        section_text: The text content of the section
        section_hierarchy: Dictionary containing the section's header hierarchy

    Returns:
        A summary of the section's content
    """
    # Extract first sentence or paragraph for a simple summary
    first_paragraph = section_text.split('\n\n')[0] if '\n\n' in section_text else section_text
    first_sentence = first_paragraph.split('. ')[0] if '. ' in first_paragraph else first_paragraph

    # Combine headers with first sentence for context
    header_context = " > ".join(section_hierarchy.values()) if section_hierarchy else ""

    # Limit summary length
    max_summary_length = 200
    summary = first_sentence[:max_summary_length]
    if len(first_sentence) > max_summary_length:
        summary += "..."

    if header_context:
        return f"{header_context}: {summary}"
    return summary


def split_text_with_overlap(text, chunk_size=config.chunk_size, overlap=config.overlap, max_sentence_words=100):
    """
    Simplified method to split text into chunks with overlap.
    Uses approximate character counts instead of tokenization.
    """
    paragraphs = text.split("# ")
    chunks = []
    chunk_metadata = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        sentences = sent_tokenize(paragraph)
        current_chunk = ""
        current_length = 0
        # Approximation: 4 characters ≈ 1 token
        char_chunk_size = chunk_size * 4
        char_overlap = overlap * 4

        for sentence in sentences:
            if is_mostly_html(sentence):
                continue

            words = sentence.split()

            # Handle long sentences
            if len(words) > max_sentence_words:
                logger.debug(f"Splitting long sentence...")
                logger.debug(f"{sentence[:50]}...")
                sub_sentences = re.split(r'(?<=[,;]) ', sentence)  # Split at commas/semicolons

                if len(sub_sentences) == 1:  # If no good split, do word-based splitting
                    sub_sentences = [
                        " ".join(words[i: i + max_sentence_words])
                        for i in range(0, len(words), max_sentence_words)
                    ]

                # Process each sub-sentence separately
                for sub_sentence in sub_sentences:
                    sentence_length = len(sub_sentence)

                    if current_length + sentence_length > char_chunk_size:
                        # Save the current chunk
                        if current_chunk:
                            chunks.append(current_chunk)
                            chunk_metadata.append({"section_hierarchy": {}, "section_summary": ""})

                            # Retain overlap from previous chunk
                            if char_overlap > 0:
                                words_for_overlap = current_chunk.split(" ")
                                overlap_text = " ".join(
                                    words_for_overlap[-int(char_overlap / 4):])  # Approximate word count
                                current_chunk = overlap_text
                                current_length = len(current_chunk)
                            else:
                                current_chunk = ""
                                current_length = 0

                    current_chunk += " " + sub_sentence if current_chunk else sub_sentence
                    current_length += sentence_length
            else:
                # Regular sentence processing
                sentence_length = len(sentence)

                if current_length + sentence_length > char_chunk_size:
                    # Save the current chunk
                    if current_chunk:
                        chunks.append(current_chunk)
                        chunk_metadata.append({"section_hierarchy": {}, "section_summary": ""})

                        # Retain overlap from previous chunk
                        if char_overlap > 0:
                            words_for_overlap = current_chunk.split(" ")
                            overlap_text = " ".join(
                                words_for_overlap[-int(char_overlap / 4):])  # Approximate word count
                            current_chunk = overlap_text
                            current_length = len(current_chunk)
                        else:
                            current_chunk = ""
                            current_length = 0

                current_chunk += " " + sentence if current_chunk else sentence
                current_length += sentence_length

        # Save the last chunk if it has content
        if current_chunk:
            chunks.append(current_chunk)
            chunk_metadata.append({"section_hierarchy": {}, "section_summary": ""})

    return chunks, chunk_metadata