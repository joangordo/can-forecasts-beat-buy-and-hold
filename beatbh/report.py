"""Console/file logging, CSV tables and the auto-generated results section."""
import logging
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .config import ROOT

log = logging.getLogger("beatbh")
README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


@contextmanager
def quiet():
    previous = log.disabled
    log.disabled = True
    try:
        yield
    finally:
        log.disabled = previous


class Reporter:
    """Writes every table to CSV and prints it; mirrors the console to full_report.txt."""

    def __init__(self, out_dir):
        self.out = Path(out_dir)
        self.fig_dir, self.tab_dir = self.out / "figures", self.out / "tables"
        for d in (self.out, self.fig_dir, self.tab_dir):
            d.mkdir(parents=True, exist_ok=True)
        for old in [*self.fig_dir.glob("*.png"), *self.tab_dir.glob("*.csv")]:
            old.unlink()  # never mix outputs of different runs
        log.setLevel(logging.INFO)
        log.handlers.clear()
        log.propagate = False
        for handler in (logging.StreamHandler(),
                        logging.FileHandler(self.out / "full_report.txt", mode="w", encoding="utf-8")):
            handler.setFormatter(logging.Formatter("%(message)s"))
            log.addHandler(handler)

    def section(self, title):
        log.info("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)

    def text(self, message=""):
        log.info(message)

    def table(self, df, name, decimals=3, show=True):
        df.round(6).to_csv(self.tab_dir / f"{name}.csv")
        if show:
            log.info(df.to_string(float_format=lambda v: f"{v:,.{decimals}f}"))


def _cell(value, decimals):
    if isinstance(value, (float, np.floating)):
        return "–" if np.isnan(value) else f"{value:,.{decimals}f}"
    return str(value)


def md_table(df, decimals=2):
    head = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [_cell(v, decimals) for v in row.values]) + " |")
    return "\n".join(lines)


def inject_readme(markdown):
    """Replace the block between the RESULTS markers of README.md."""
    readme = ROOT / "README.md"
    if not readme.exists():
        return False
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(README_START) + ".*?" + re.escape(README_END), re.S)
    if not pattern.search(text):
        return False
    readme.write_text(pattern.sub(lambda _: f"{README_START}\n{markdown}\n{README_END}", text),
                      encoding="utf-8")
    return True
