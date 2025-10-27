"""Tests for the coverage badge generation helper."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _script_path() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "generate_coverage_badge.py"


def test_default_output_is_alongside_coverage_xml(tmp_path):
    """The badge should live next to the coverage XML when no output is provided."""

    coverage_xml = tmp_path / "reports" / "coverage.xml"
    coverage_xml.parent.mkdir(parents=True)
    coverage_xml.write_text(
        """<?xml version='1.0'?>\n<coverage line-rate='0.83' branch-rate='0.5'></coverage>""",
        encoding="utf-8",
    )

    working_directory = tmp_path / "working"
    working_directory.mkdir()

    subprocess.run(
        [sys.executable, str(_script_path()), str(coverage_xml)],
        check=True,
        cwd=working_directory,
    )

    badge = coverage_xml.with_name("coverage.svg")
    assert badge.is_file(), "Badge should be generated next to the coverage XML"
    assert not (working_directory / "coverage.svg").exists(), "Badge should not be written to CWD"

    svg_contents = badge.read_text(encoding="utf-8")
    assert "coverage" in svg_contents
    assert "83%" in svg_contents
