"""
parse_html.py
Converts Moodle forum HTML exports to the JSON format used by the summarizer.

Output mirrors the structure of tests/forum_d06_ethics.json:
  {
    "title": "...",
    "course": "...",
    "posts": [
      {
        "id": "p...",
        "author": "...",
        "date": "ISO-8601 string",
        "subject": "...",
        "content": "plain text",
        "replies": [ ...recursive same shape... ]
      }
    ]
  }

Usage:
    python parse_html.py                          # default: data-sample/ -> parsed/
    python parse_html.py --input data-sample/ --output parsed/
"""

import re
import json
import argparse
from pathlib import Path
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_author(name: str) -> str:
    """Strip Moodle student-ID suffix from display names.

    Moodle shows students as "Full Name - 2306275046 LastName".
    Lecturers have no suffix ("Syifa Nurhayati Syifa").
    """
    return re.sub(r'\s*-\s*\d{6,12}\s+\S+\s*$', '', name).strip()


def clean_text(element) -> str:
    """Return normalised plain text from a BS4 element."""
    if element is None:
        return ""
    text = element.get_text(separator=' ', strip=True)
    return re.sub(r'\s+', ' ', text).strip()


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_post(article) -> dict:
    """Recursively parse one <article data-region="post"> element."""
    post_id = article.get('data-post-id', '')

    # The actual post body lives in the sibling div with data-content="forum-post"
    # (direct child of <article>, distinct from the replies-container)
    forumpost_div = article.find(
        'div', attrs={'data-content': 'forum-post'}, recursive=False
    )
    if forumpost_div is None:
        forumpost_div = article  # fallback

    # --- header fields ---
    header = forumpost_div.find('header')
    author = date = subject = ""

    if header:
        # Author: the <a> that links to the user profile page
        author_link = header.find(
            'a', href=lambda h: h and '/user/view.php' in h
        )
        if author_link:
            author = clean_author(author_link.get_text(strip=True))

        time_tag = header.find('time')
        if time_tag:
            date = time_tag.get('datetime', '')

        subject_tag = header.find(
            'h3', attrs={'data-region-content': 'forum-post-core-subject'}
        )
        if subject_tag:
            subject = subject_tag.get_text(strip=True)

    # --- content ---
    content_div = forumpost_div.find('div', id=f'post-content-{post_id}')
    # For the root (firstpost) Moodle wraps the real content inside a nested
    # post-content-container. Using get_text() on the outer div captures it all.
    content = clean_text(content_div)

    # --- replies (direct children only via the indent div) ---
    replies_container = article.find(
        'div', attrs={'data-region': 'replies-container'}, recursive=False
    )
    replies = []
    if replies_container:
        for child in replies_container.find_all(
            'article', attrs={'data-region': 'post'}, recursive=False
        ):
            replies.append(parse_post(child))

    return {
        "id": f"p{post_id}",
        "author": author,
        "date": date,
        "subject": subject,
        "content": content,
        "replies": replies,
    }


def parse_html_file(html_path: Path, title: str = "", course: str = "") -> dict:
    """Parse one Moodle forum HTML file into the summarizer JSON format."""
    html = html_path.read_text(encoding='utf-8', errors='replace')
    soup = BeautifulSoup(html, 'html.parser')

    # The root post is the very first article with data-region="post"
    root_article = soup.find('article', attrs={'data-region': 'post'})
    if root_article is None:
        return {"title": title, "course": course, "posts": []}

    root_post = parse_post(root_article)

    if not title:
        title = root_post.get('subject', html_path.stem)

    return {
        "title": title,
        "course": course,
        "posts": [root_post],
    }


# ---------------------------------------------------------------------------
# Directory walker
# ---------------------------------------------------------------------------

LABEL_MAP = {
    'sister-gasal-20232024': 'SISTER Gasal 2023/2024',
    'sister-gasal-20242025': 'SISTER Gasal 2024/2025',
    'sister-genap-20252026': 'SISTER Genap 2025/2026',
    'FD': 'Forum Diskusi',
    'WR': 'Weekly Reflection',
}


def derive_course(input_dir: Path, rel: Path) -> str:
    """Build a human-readable course label from the relative path parts."""
    # rel.parts[:-1] gives the directory components (drop the filename).
    # If the file lives at the root of input_dir there are no parts, so fall
    # back to the input_dir name itself.
    dir_parts = list(rel.parts[:-1])
    parts = dir_parts if dir_parts else [input_dir.name]
    return ' / '.join(LABEL_MAP.get(p, p) for p in parts)


def process_directory(input_dir: Path, output_dir: Path):
    html_files = sorted(input_dir.rglob('*.html'))
    print(f"Found {len(html_files)} HTML file(s) under '{input_dir}'")

    for html_path in html_files:
        rel = html_path.relative_to(input_dir)
        out_path = (output_dir / rel).with_suffix('.json')
        out_path.parent.mkdir(parents=True, exist_ok=True)

        course = derive_course(input_dir, rel)
        print(f"  Parsing {rel} ...", end=' ', flush=True)

        result = parse_html_file(html_path, course=course)
        post_count = sum(1 + _count_replies(p) for p in result['posts'])
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        print(f"saved ({post_count} post/reply nodes) -> {out_path.relative_to(output_dir.parent)}")

    print(f"\nDone. JSONs written to '{output_dir}'")


def _count_replies(post: dict) -> int:
    return sum(1 + _count_replies(r) for r in post.get('replies', []))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Parse Moodle forum HTML exports to summarizer-compatible JSON'
    )
    parser.add_argument(
        '--input', default='data-sample',
        help='Root directory containing HTML files (default: data-sample)'
    )
    parser.add_argument(
        '--output', default='parsed',
        help='Output directory for JSON files (default: parsed)'
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: input directory '{input_dir}' does not exist.")
        raise SystemExit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    process_directory(input_dir, output_dir)


if __name__ == '__main__':
    main()
