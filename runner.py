"""Clean, parse and extract structured data from raw mailing-list text exports.

Reads .txt files (mbox-like or single-message) from a data directory, extracts
RFC headers where available, cleans the body from quoted replies/diffs and
system paths, extracts common DPDK tags (Acked-by, Reviewed-by, ...), and
writes a cleaned CSV suitable for NLP or analytics.

Usage:
  python runner.py [--data-dir DIR] [--output FILE]
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from email import policy
from email.parser import Parser
from email.header import decode_header
from email.utils import parseaddr
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
except Exception:
    pd = None


# Defaults
DATA_DIR = os.path.dirname(__file__) or "."
DEFAULT_OUTPUT = os.path.join(DATA_DIR, "cleaned_emails.csv")


# DPDK tags and regexes
DPDK_TAGS = [
    "Acked-by",
    "Reviewed-by",
    "Nacked-by",
    "Signed-off-by",
    "Tested-by",
    "Suggested-by",
    "Reported-by",
]
DPDK_TAG_RE = re.compile(r"^(?P<tag>" + "|".join([re.escape(t) for t in DPDK_TAGS]) + r")\s*:\s*(?P<value>.+)", re.I)
ON_HEADER_RE = re.compile(r"^On\s.+wrote:\s*", re.I)
DIFF_RE = re.compile(r"^diff --git", re.M)
SYSTEM_PATH_RE = re.compile(r"^([A-Za-z]:)?[\\/].+\.[A-Za-z0-9]{1,5}$")


def read_text_files(path_pattern: str) -> List[str]:
    files = glob.glob(path_pattern)
    contents: List[str] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                contents.append(f.read())
        except Exception:
            with open(fp, "r", encoding="latin-1", errors="replace") as f:
                contents.append(f.read())
    return contents


def split_mbox_like(raw: str) -> List[str]:
    lines = raw.splitlines(keepends=True)
    msgs: List[str] = []
    cur: List[str] = []
    for ln in lines:
        if ln.startswith("From ") and re.search(r"\d{4}", ln):
            if cur:
                msgs.append("".join(cur))
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        msgs.append("".join(cur))
    return msgs


def _decode_mime_header(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    parts = []
    for chunk, enc in decode_header(value):
        if isinstance(chunk, bytes):
            try:
                parts.append(chunk.decode(enc or "utf-8", errors="replace"))
            except Exception:
                parts.append(chunk.decode("utf-8", errors="replace"))
        else:
            parts.append(chunk)
    return "".join(parts).strip()


def extract_dpdk_tags(text: str) -> Dict[str, List[str]]:
    tags: Dict[str, List[str]] = {}
    for line in text.splitlines():
        m = DPDK_TAG_RE.match(line.strip())
        if m:
            tag = m.group("tag").strip()
            val = m.group("value").strip()
            tags.setdefault(tag, []).append(val)
    return tags


def clean_body(text: Optional[str]) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    m = DIFF_RE.search(text)
    if m:
        text = text[: m.start()]
    out_lines: List[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            out_lines.append("")
            continue
        if s.startswith(">"):
            continue
        if ON_HEADER_RE.match(s):
            continue
        if SYSTEM_PATH_RE.match(s):
            continue
        if re.match(r"^\S+\s+\|\s*\d+", s):
            continue
        out_lines.append(ln)
    cleaned = "\n".join(out_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_email_parts(raw_msg: str) -> Dict[str, Optional[str]]:
    parser = Parser(policy=policy.default)
    try:
        em = parser.parsestr(raw_msg)
    except Exception:
        em = None

    def hdr(k: str) -> Optional[str]:
        if em and em[k]:
            return _decode_mime_header(str(em[k]))
        m = re.search(rf"^{k}:\s*(.+)$", raw_msg, re.I | re.M)
        return m.group(1).strip() if m else None

    from_hdr = hdr("From")
    to_hdr = hdr("To")
    date_hdr = hdr("Date")
    subject_hdr = hdr("Subject")

    body = None
    if em is not None:
        try:
            if em.is_multipart():
                parts: List[str] = []
                for part in em.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if isinstance(payload, bytes):
                            parts.append(payload.decode(part.get_content_charset() or "utf-8", errors="replace"))
                        else:
                            parts.append(str(payload))
                body = "\n".join(parts)
            else:
                payload = em.get_payload(decode=True)
                if isinstance(payload, bytes):
                    body = payload.decode(em.get_content_charset() or "utf-8", errors="replace")
                else:
                    body = str(em.get_payload())
        except Exception:
            body = None

    if not body:
        parts = re.split(r"\r?\n\r?\n", raw_msg, maxsplit=1)
        body = parts[1] if len(parts) > 1 else (parts[0] if parts else "")

    tags = extract_dpdk_tags(body)
    clean = clean_body(body)
    return {"from": from_hdr, "to": to_hdr, "date": date_hdr, "subject": subject_hdr, "body": clean, "tags": tags}


def normalize_sender(sender: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not sender:
        return None, None
    decoded = _decode_mime_header(sender) or sender
    name, addr = parseaddr(decoded)
    name = name.strip() or None
    addr = addr or None
    return name, addr


def process_all_files(data_dir: str):
    pattern = os.path.join(data_dir, "*.txt")
    raws = read_text_files(pattern)
    if not raws:
        print("No .txt files found in the data directory; using an example message.")
        raws = [
            (
                "From: Alice Example <alice@example.com>\n"
                "To: dpdk@dpdk.org\n"
                "Date: Mon, 05 Mar 2018 12:34:56 +0000\n"
                "Subject: [PATCH] example\n\n"
                "This is a small example message.\n\n"
                "Acked-by: Bob Reviewer <bob@example.com>\n"
            )
        ]

    messages: List[Dict[str, Optional[str]]] = []
    for raw in raws:
        for part in split_mbox_like(raw):
            parsed = extract_email_parts(part)
            name, email_addr = normalize_sender(parsed.get("from"))
            messages.append({
                "from_raw": parsed.get("from"),
                "from_name": name,
                "from_email": email_addr,
                "to": parsed.get("to"),
                "date": parsed.get("date"),
                "subject": parsed.get("subject"),
                "body": parsed.get("body"),
                "tags": parsed.get("tags"),
            })

    df = pd.DataFrame(messages) if pd is not None else None
    if df is None:
        raise RuntimeError("pandas is required to run this script. Install with: pip install pandas")

    for tag in DPDK_TAGS:
        df[tag] = df["tags"].apply(lambda d, t=tag: " || ".join(d.get(t, [])) if isinstance(d, dict) else "")

    df = df.drop(columns=["tags"]) if "tags" in df.columns else df
    return df


def write_csv(df, out_path: str) -> None:
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parse and clean mailing-list .txt exports into CSV")
    p.add_argument("--data-dir", default=DATA_DIR, help="Directory containing .txt files")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    df = process_all_files(args.data_dir)
    write_csv(df, args.output)
    print(f"Wrote {len(df)} messages to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
runner.py

A focused, robust script to parse raw email text files (DPDK mailing list exports),
extract headers and body, clean quoted replies, diffs, file paths, and extract
DPDK metadata like Acked-by, Reviewed-by, Nacked-by. Normalizes senders and
outputs a clean CSV suitable for NLP.

Usage:
  - Place raw .txt email files in the same directory (or update DATA_DIR).
  - Run: python runner.py

Output: cleaned_emails.csv

"""

from __future__ import annotations
import re
import os
import csv
import sys
import email
import glob
from typing import List, Dict, Tuple, Optional
import pandas as pd

# Configuration
DATA_DIR = os.path.dirname(__file__) or "."
OUTPUT_CSV = os.path.join(DATA_DIR, "cleaned_emails.csv")
MAIL_GLOB = os.path.join(DATA_DIR, "*.txt")

# Patterns
DPDK_TAGS = ["Acked-by", "Reviewed-by", "Nacked-by", "Signed-off-by", "Tested-by", "Suggested-by", "Reported-by"]
DPDK_TAG_RE = re.compile(r"^(?P<tag>" + "|".join([re.escape(t) for t in DPDK_TAGS]) + r")\s*:\s*(?P<value>.+)", re.I)
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.[A-Za-z]{2,}")
ON_HEADER_RE = re.compile(r"^On\s.+wrote:\\s*", re.I)
DIFF_RE = re.compile(r"^diff --git", re.M)
SYSTEM_PATH_RE = re.compile(r"^[\w\d\s\-_/\\]+/[\w\d\s\-_/\\]+")
FILE_INS_DEL_RE = re.compile(r".*/[\w\d\-_]+\.[a-z]{1,}\s*\|\s*\d+\s*[\+\-]*")

# Utility functions

def read_text_files(path_pattern: str) -> List[str]:
    files = glob.glob(path_pattern)
    contents = []
    if not files:
        return contents
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                contents.append(f.read())
        except Exception:
            # try latin-1 as fallback
            with open(fp, "r", encoding="latin-1", errors="replace") as f:
                contents.append(f.read())
    return contents


def split_mbox_like(raw: str) -> List[str]:
    """Split a mbox-like text into messages by 'From ' lines that look like senders.
    Falls back to the entire file as one message if no split found."""
    lines = raw.splitlines(keepends=True)
    msgs = []
    cur = []
    for ln in lines:
        if ln.startswith("From ") and re.search(r"\d{4}", ln):
            if cur:
                msgs.append("".join(cur))
                cur = [ln]
            else:
                cur = [ln]
        else:
            cur.append(ln)
    if cur:
        msgs.append("".join(cur))
    return msgs


def extract_email_parts(raw_msg: str) -> Dict[str, Optional[str]]:
    """Attempt to extract headers and a cleaned body from a raw text message.

    Returns a dict with keys: from, to, date, subject, body, tags (dict)
    """
    result = {"from": None, "to": None, "date": None, "subject": None, "body": None, "tags": {}}

    # Try to parse with the email library first (if message has RFC headers)
    try:
        em = email.message_from_string(raw_msg)
    except Exception:
        em = None

    # Headers
    def get_hdr(k: str) -> Optional[str]:
        if em and em.get(k):
            return em.get(k)
        # fallback: simple regex
        m = re.search(rf"^{k}:\s*(.+)$", raw_msg, re.I | re.M)
        return m.group(1).strip() if m else None

    result["from"] = get_hdr("From")
    result["to"] = get_hdr("To")
    result["date"] = get_hdr("Date")
    result["subject"] = get_hdr("Subject")

    # body extraction: if email lib provided payload, prefer it
    body = None
    if em:
        try:
            if em.is_multipart():
                parts = []
                for part in em.walk():
                    if part.get_content_type() == "text/plain":
                        try:
                            parts.append(part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace"))
                        except Exception:
                            parts.append(part.get_payload())
                body = "\n".join(parts)
            else:
                payload = em.get_payload(decode=True)
                if payload is None:
                    body = em.get_payload()
                else:
                    try:
                        body = payload.decode(em.get_content_charset() or "utf-8", errors="replace")
                    except Exception:
                        body = payload.decode("utf-8", errors="replace")
        except Exception:
            body = None

    """runner.py

    Clean, parse and extract structured data from raw mailing-list text exports
    (mbox-style or single-message .txt files). Produces a `cleaned_emails.csv` by
    default.

    What it does:
    - Reads .txt files from a data directory (default: the script directory).
    - Splits mbox-like files into messages when possible.
    - Extracts From/To/Date/Subject and the plain text body.
    - Cleans quoted replies, diffs, system paths and other noisy lines.
    - Extracts common DPDK tags (Acked-by, Reviewed-by, Nacked-by, ...).
    - Normalizes sender name/email and writes a CSV ready for NLP.

    Usage:
      python runner.py [--data-dir DIR] [--output FILE]

    """

    from __future__ import annotations
    import argparse
    import csv
    import glob
    import os
    import re
    import sys
    from email import policy
    from email.parser import Parser
    from email.header import decode_header
    from email.utils import parseaddr
    from typing import Dict, List, Optional, Tuple

    try:
        import pandas as pd
    except Exception:  # pragma: no cover - graceful fallback for environments without pandas
        pd = None

    # Configuration defaults
    DATA_DIR = os.path.dirname(__file__) or "."
    DEFAULT_OUTPUT = os.path.join(DATA_DIR, "cleaned_emails.csv")
    MAIL_GLOB = os.path.join(DATA_DIR, "*.txt")

    # DPDK-ish tags we want to capture (case-insensitive)
    DPDK_TAGS = [
        "Acked-by",
        "Reviewed-by",
        "Nacked-by",
        "Signed-off-by",
        "Tested-by",
        "Suggested-by",
        "Reported-by",
    ]
    DPDK_TAG_RE = re.compile(r"^(?P<tag>" + "|".join([re.escape(t) for t in DPDK_TAGS]) + r")\s*:\s*(?P<value>.+)", re.I)

    # Useful regexes for cleaning
    ON_HEADER_RE = re.compile(r"^On\s.+wrote:\s*", re.I)
    DIFF_RE = re.compile(r"^diff --git", re.M)
    SYSTEM_PATH_RE = re.compile(r"^([A-Za-z]:)?[\\/].+\.[A-Za-z0-9]{1,5}$")
    EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.[A-Za-z]{2,}")


    def read_text_files(path_pattern: str) -> List[str]:
        files = glob.glob(path_pattern)
        contents: List[str] = []
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    contents.append(f.read())
            except Exception:
                with open(fp, "r", encoding="latin-1", errors="replace") as f:
                    contents.append(f.read())
        return contents


    def split_mbox_like(raw: str) -> List[str]:
        """Split a mbox-like text into messages by lines beginning with 'From '.

        If no mbox separators appear, return the original text as a single message.
        """
        lines = raw.splitlines(keepends=True)
        msgs: List[str] = []
        cur: List[str] = []
        for ln in lines:
            if ln.startswith("From ") and re.search(r"\d{4}", ln):
                if cur:
                    msgs.append("".join(cur))
                cur = [ln]
            else:
                cur.append(ln)
        if cur:
            msgs.append("".join(cur))
        return msgs


    def _decode_mime_header(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        parts = []
        for chunk, enc in decode_header(value):
            if isinstance(chunk, bytes):
                try:
                    parts.append(chunk.decode(enc or "utf-8", errors="replace"))
                except Exception:
                    parts.append(chunk.decode("utf-8", errors="replace"))
            else:
                parts.append(chunk)
        return "".join(parts).strip()


    def extract_dpdk_tags(text: str) -> Dict[str, List[str]]:
        tags: Dict[str, List[str]] = {}
        for line in text.splitlines():
            m = DPDK_TAG_RE.match(line.strip())
            if m:
                tag = m.group("tag").strip()
                val = m.group("value").strip()
                tags.setdefault(tag, []).append(val)
        return tags


    def clean_body(text: Optional[str]) -> str:
        """Remove quoted replies, diffs, system paths and tidy whitespace."""
        if not text:
            return ""
        # normalize newlines
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # cut off at the first diff marker (common long noise)
        m = DIFF_RE.search(text)
        if m:
            text = text[: m.start()]

        out_lines: List[str] = []
        for ln in text.splitlines():
            s = ln.strip()
            if not s:
                out_lines.append("")
                continue
            if s.startswith(">"):
                continue
            if ON_HEADER_RE.match(s):
                continue
            # skip obvious single-file lines and system paths
            if SYSTEM_PATH_RE.match(s):
                continue
            # drop typical patch summary lines
            if re.match(r"^\S+\s+\|\s*\d+", s):
                continue
            out_lines.append(ln)

        cleaned = "\n".join(out_lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()


    def extract_email_parts(raw_msg: str) -> Dict[str, Optional[str]]:
        """Return a dict with keys: from, to, date, subject, body, tags"""
        # use email.Parser with default policy for robust parsing
        parser = Parser(policy=policy.default)
        try:
            em = parser.parsestr(raw_msg)
        except Exception:
            em = None

        def hdr(k: str) -> Optional[str]:
            if em and em[k]:
                return _decode_mime_header(str(em[k]))
            m = re.search(rf"^{k}:\s*(.+)$", raw_msg, re.I | re.M)
            return m.group(1).strip() if m else None

        from_hdr = hdr("From")
        to_hdr = hdr("To")
        date_hdr = hdr("Date")
        subject_hdr = hdr("Subject")

        # body extraction (prefer library payload)
        body = None
        if em is not None:
            try:
                if em.is_multipart():
                    parts: List[str] = []
                    for part in em.walk():
                        if part.get_content_type() == "text/plain":
                            payload = part.get_payload(decode=True)
                            if isinstance(payload, bytes):
                                parts.append(payload.decode(part.get_content_charset() or "utf-8", errors="replace"))
                            else:
                                parts.append(str(payload))
                    body = "\n".join(parts)
                else:
                    payload = em.get_payload(decode=True)
                    if isinstance(payload, bytes):
                        body = payload.decode(em.get_content_charset() or "utf-8", errors="replace")
                    else:
                        body = str(em.get_payload())
            except Exception:
                body = None

        if not body:
            parts = re.split(r"\r?\n\r?\n", raw_msg, maxsplit=1)
            body = parts[1] if len(parts) > 1 else (parts[0] if parts else "")

        tags = extract_dpdk_tags(body)
        clean = clean_body(body)

        return {
            "from": from_hdr,
            "to": to_hdr,
            "date": date_hdr,
            "subject": subject_hdr,
            "body": clean,
            "tags": tags,
        }
if __name__ == "__main__":
    sys.exit(main())
