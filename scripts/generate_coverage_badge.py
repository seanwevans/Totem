"""Generate a simple SVG coverage badge from a coverage XML report."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "coverage_xml",
        help="Path to the coverage XML report produced by `coverage xml`.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="coverage.svg",
        help="Destination path for the generated badge (default: coverage.svg).",
    )
    return parser.parse_args()


def load_coverage_percentage(path: str) -> int:
    try:
        tree = ET.parse(path)
    except FileNotFoundError as exc:  # pragma: no cover - handled by argument parsing
        raise SystemExit(f"coverage file not found: {path}") from exc
    except ET.ParseError as exc:  # pragma: no cover - indicates malformed XML
        raise SystemExit(f"failed to parse coverage XML: {exc}") from exc

    root = tree.getroot()
    try:
        rate = float(root.attrib["line-rate"])
    except KeyError as exc:  # pragma: no cover - indicates unexpected XML schema
        raise SystemExit("coverage XML is missing the 'line-rate' attribute") from exc

    return int(round(rate * 100))


def badge_color(percentage: int) -> str:
    if percentage >= 90:
        return "#4c1"  # bright green
    if percentage >= 80:
        return "#97CA00"  # green
    if percentage >= 70:
        return "#a4a61d"  # yellow-green
    if percentage >= 60:
        return "#dfb317"  # yellow
    if percentage >= 50:
        return "#fe7d37"  # orange
    return "#e05d44"  # red


def value_width(value: str) -> int:
    base = 36
    extra = max(0, len(value) - 3) * 7
    return base + extra


def render_svg(percentage: int) -> str:
    label = "coverage"
    value = f"{percentage}%"

    label_width = 63
    value_width_px = value_width(value)
    total_width = label_width + value_width_px

    label_center = label_width / 2
    value_center = label_width + value_width_px / 2

    color = badge_color(percentage)

    return f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{total_width}\" height=\"20\"> 
    <linearGradient id=\"b\" x2=\"0\" y2=\"100%\">
        <stop offset=\"0\" stop-color=\"#bbb\" stop-opacity=\".1\"/>
        <stop offset=\"1\" stop-opacity=\".1\"/>
    </linearGradient>
    <mask id=\"a\">
        <rect width=\"{total_width}\" height=\"20\" rx=\"3\" fill=\"#fff\"/>
    </mask>
    <g mask=\"url(#a)\">
        <path fill=\"#555\" d=\"M0 0h{label_width}v20H0z\"/>
        <path fill=\"{color}\" d=\"M{label_width} 0h{value_width_px}v20H{label_width}z\"/>
        <path fill=\"url(#b)\" d=\"M0 0h{total_width}v20H0z\"/>
    </g>
    <g fill=\"#fff\" text-anchor=\"middle\" font-family=\"DejaVu Sans,Verdana,Geneva,sans-serif\" font-size=\"11\">
        <text x=\"{label_center}\" y=\"15\" fill=\"#010101\" fill-opacity=\".3\">{label}</text>
        <text x=\"{label_center}\" y=\"14\">{label}</text>
        <text x=\"{value_center}\" y=\"15\" fill=\"#010101\" fill-opacity=\".3\">{value}</text>
        <text x=\"{value_center}\" y=\"14\">{value}</text>
    </g>
</svg>
"""


def main() -> int:
    args = parse_args()
    percentage = load_coverage_percentage(args.coverage_xml)
    svg = render_svg(percentage)

    args.output = args.output.strip()
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(svg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
