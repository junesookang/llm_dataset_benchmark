# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
"""Download Paul Graham essays and bundle them into a single json payload."""

import glob
import json
import logging
import shutil
from contextlib import suppress
from pathlib import Path
from typing import Iterable, Tuple
from urllib.error import URLError
from urllib.request import urlopen

import html2text
from bs4 import BeautifulSoup
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
URL_LIST_PATH = SCRIPT_DIR / "PaulGrahamEssays_URLs.txt"
OUTPUT_JSON = SCRIPT_DIR / "PaulGrahamEssays.json"
REPO_DIR = SCRIPT_DIR / "essay_repo"
HTML_DIR = SCRIPT_DIR / "essay_html"

logging.basicConfig(level=logging.INFO, force=True)
LOGGER = logging.getLogger(__name__)


def configure_html2text() -> html2text.HTML2Text:
    """Return a configured html2text converter."""
    converter = html2text.HTML2Text()
    converter.ignore_images = True
    converter.ignore_tables = True
    converter.escape_all = True
    converter.reference_links = False
    converter.mark_code = False
    return converter


def read_url_list(path: Path) -> Iterable[str]:
    """Read newline-delimited URL list."""
    with path.open("r", encoding="utf-8") as url_file:
        for line in url_file:
            stripped = line.strip()
            if stripped:
                yield stripped


def fetch_content(url: str) -> Tuple[str, bytes]:
    """Download content from URL and return filename and raw bytes."""
    filename = url.split("/")[-1]
    with urlopen(url) as response:
        return filename, response.read()


def write_text_file(directory: Path, filename: str, content: str) -> None:
    """Persist text content to the target directory."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / filename).open("w", encoding="utf-8") as file:
        file.write(content)


def process_html(url: str, converter: html2text.HTML2Text) -> None:
    """Download an HTML page and save converted text to disk."""
    filename_raw, content = fetch_content(url)
    soup = BeautifulSoup(content.decode("unicode_escape", "utf-8"), "html.parser")
    specific_tag = soup.find("font")
    if specific_tag is None:
        raise ValueError(f"No <font> tag found in {url}")
    parsed = converter.handle(str(specific_tag))
    filename = filename_raw.replace(".html", ".txt")
    write_text_file(HTML_DIR, filename, parsed)


def process_plain_text(url: str) -> None:
    """Download a plain text file from URL."""
    filename, content = fetch_content(url)
    write_text_file(REPO_DIR, filename, content.decode("utf-8"))


def combine_text_files(patterns: Iterable[str]) -> str:
    """Concatenate content from glob patterns into a single string."""
    combined = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path, "r", encoding="utf-8") as file:
                combined.append(file.read())
    return "".join(combined)


def clean_temp_directories() -> None:
    """Remove working directories used for downloads."""
    for directory in (REPO_DIR, HTML_DIR):
        with suppress(FileNotFoundError):
            shutil.rmtree(directory)


def main() -> None:
    converter = configure_html2text()
    urls = list(read_url_list(URL_LIST_PATH))

    for url in tqdm(urls, desc="Downloading essays"):
        try:
            if url.endswith(".html"):
                process_html(url, converter)
            else:
                process_plain_text(url)
        except URLError as exc:
            LOGGER.warning("Network error downloading %s: %s", url, exc)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Failed to process %s: %s", url, exc)

    repo_files = sorted(glob.glob(str(REPO_DIR / "*.txt")))
    html_files = sorted(glob.glob(str(HTML_DIR / "*.txt")))
    LOGGER.info(
        "Downloaded %d essays from https://github.com/gkamradt/LLMTest_NeedleInAHaystack/",
        len(repo_files),
    )
    LOGGER.info(
        "Downloaded %d essays from http://www.paulgraham.com/",
        len(html_files),
    )

    combined_text = combine_text_files(
        [str(REPO_DIR / "*.txt"), str(HTML_DIR / "*.txt")]
    )
    with OUTPUT_JSON.open("w", encoding="utf-8") as output_file:
        json.dump({"text": combined_text}, output_file)

    clean_temp_directories()


if __name__ == "__main__":
    main()
