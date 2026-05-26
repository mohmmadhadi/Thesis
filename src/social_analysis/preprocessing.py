"""Shared text preprocessing utilities extracted from the notebooks."""

from __future__ import annotations

import re
from typing import Any


EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251]+",
    flags=re.UNICODE,
)


class TextPreprocessor:
    """Clean tweet-like text using the shared notebook preprocessing rules."""

    def __init__(
        self,
        keep_emojis: bool = True,
        remove_urls: bool = True,
        lowercase: bool = True,
    ) -> None:
        self.keep_emojis = keep_emojis
        self.remove_urls = remove_urls
        self.lowercase = lowercase

    def clean_text(self, text: str) -> str:
        """Return a cleaned string while preserving notebook regex behavior."""
        if not isinstance(text, str) or text.strip() == "":
            return ""

        if self.lowercase:
            text = text.lower()

        if self.remove_urls:
            text = re.sub(r"http\S+|www\.\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#(\w+)", r"\1", text)
        text = re.sub(r"#", "", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)
        text = re.sub(r"&quot;", '"', text)
        text = re.sub(r"&#\d+;", "", text)

        if not self.keep_emojis:
            text = EMOJI_RE.sub("", text)

        return re.sub(r"\s+", " ", text).strip()

    def transform_dataframe(
        self,
        df: Any,
        text_col: str,
        output_col: str = "clean_text",
    ) -> Any:
        """Return a copy of a dataframe with a cleaned text column added."""
        result = df.copy()
        result[output_col] = result[text_col].apply(self.clean_text)
        return result
