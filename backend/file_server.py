
from pathlib import Path
import re

from mcp.server import MCPServer


# --------------------------------------------------
# MCP Server Configuration
# --------------------------------------------------

mcp = MCPServer("File Server")

NOTES_DIR = Path(__file__).parent / "data" / "notes"
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


# --------------------------------------------------
# Search for Files
# --------------------------------------------------

@mcp.tool()
def search_files(query: str) -> list[str]:
    """Search the notes folder for filenames matching a query."""

    if not NOTES_DIR.is_dir():
        return []

    query_words = [
        word.lower()
        for word in query.split()
        if word.lower() not in {
            "notes", "note", "file", "files", "my", "the",
            "find", "show", "list", "search", "for"
        }
    ]

    matches = []

    for file_path in sorted(NOTES_DIR.iterdir()):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        filename = file_path.name.lower()

        if not query_words or any(
            word in filename for word in query_words
        ):
            matches.append(file_path.name)

    return matches


# --------------------------------------------------
# Read a File
# --------------------------------------------------

@mcp.tool()
def read_file(filename: str) -> str:
    """Read a text or PDF file from the notes folder."""

    notes_root = NOTES_DIR.resolve()
    file_path = (NOTES_DIR / filename).resolve()

    # Prevent access outside the notes directory.
    try:
        file_path.relative_to(notes_root)
    except ValueError:
        return "Access denied."

    if not file_path.is_file():
        return "File not found."

    if file_path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        try:
            reader = PdfReader(file_path)
            pages = []

            for page_number, page in enumerate(
                reader.pages, start=1
            ):
                page_text = page.extract_text() or ""

                if page_text.strip():
                    pages.append(
                        f"Page {page_number}:\n{page_text}"
                    )

            return "\n\n".join(pages) or "No extractable text found."

        except Exception as error:
            return f"Could not read PDF: {error}"

    if file_path.suffix.lower() == ".txt":
        try:
            return file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            return f"Could not read text file: {error}"

    return "Unsupported file type."


# --------------------------------------------------
# Search Inside All Study Files
# --------------------------------------------------

@mcp.tool()
def search_file_content(query: str, filename: str = "") -> str:
    """
    Search the contents of PDF and TXT files.

    If filename is supplied, search only that file.
    Otherwise, search all supported files in the notes folder.
    Return the most relevant matching passages.
    """

    from pypdf import PdfReader

    if not NOTES_DIR.is_dir():
        return "The notes folder does not exist."

    # Remove common instruction words from the query.
    stop_words = {
        "what", "is", "are", "was", "were", "the", "a", "an",
        "of", "in", "on", "to", "for", "from", "my", "me",
        "explain", "describe", "define", "discuss", "tell",
        "about", "please", "notes", "note", "pdf", "file",
        "files", "and", "how", "does", "do", "can"
    }

    query_words = {
        word.lower()
        for word in re.findall(r"[a-zA-Z0-9]+", query)
        if len(word) > 1 and word.lower() not in stop_words
    }

    if not query_words:
        return "Please provide a topic to search for."

    notes_root = NOTES_DIR.resolve()

    # Search one requested file or every supported study file.
    if filename:
        candidates = [NOTES_DIR / filename]
    else:
        candidates = sorted(
            path
            for path in NOTES_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    results = []

    for file_path in candidates:
        resolved_path = file_path.resolve()

        # Prevent access outside the notes folder.
        try:
            resolved_path.relative_to(notes_root)
        except ValueError:
            continue

        if not resolved_path.is_file():
            continue

        suffix = resolved_path.suffix.lower()

        # Extract content from PDF pages.
        if suffix == ".pdf":
            try:
                reader = PdfReader(resolved_path)
            except Exception:
                continue

            content_sections = (
                (page_number, page.extract_text() or "")
                for page_number, page in enumerate(
                    reader.pages, start=1
                )
            )

        # Extract content from text files.
        elif suffix == ".txt":
            try:
                text = resolved_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue

            chunks = re.split(r"\n\s*\n", text)

            content_sections = enumerate(chunks, start=1)

        else:
            continue

        # Score every page or text section.
        for section_number, section_text in content_sections:
            if not section_text.strip():
                continue

            normalized_text = section_text.lower()

            matched_words = [
                word
                for word in query_words
                if re.search(
                    rf"\b{re.escape(word)}\b",
                    normalized_text
                )
            ]

            if not matched_words:
                continue

            score = len(matched_words) / len(query_words)

            # Give a small bonus when the complete query phrase
            # appears in the passage.
            normalized_query = " ".join(query.lower().split())

            if normalized_query and normalized_query in normalized_text:
                score += 0.5

            results.append({
                "score": score,
                "filename": resolved_path.name,
                "section": section_number,
                "text": section_text.strip()[:1800],
            })

    if not results:
        return "No matching content found in the available notes."

    # Best matches first; retain a stable order for equal scores.
    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    selected_results = results[:5]

    return "\n\n---\n\n".join(
        f"File: {item['filename']}\n"
        f"Page/section: {item['section']}\n"
        f"Match score: {item['score']:.2f}\n"
        f"{item['text']}"
        for item in selected_results
    )
