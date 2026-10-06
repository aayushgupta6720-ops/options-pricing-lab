"""Pages made of tabs, where each tab is a script in views/sections/.

Only the open tab's script runs: a tab that isn't being looked at costs nothing, which matters on
a small server (the Convergence tab alone prices 70 trees and 200,000 paths). The open tab is kept
in the URL (?tab=Greeks), so a link can point straight at one.
"""

import runpy
from pathlib import Path

import streamlit as st

SECTIONS = Path(__file__).resolve().parents[1] / "views" / "sections"


def tabs(sections: dict[str, str]) -> None:
    """sections maps each tab's label to its script's file name in views/sections/."""
    containers = st.tabs(list(sections), key="tab", bind="query-params")
    for container, script in zip(containers, sections.values(), strict=True):
        if container.open:
            with container:
                runpy.run_path(str(SECTIONS / script))
