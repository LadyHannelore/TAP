"""
utility.py
This file contains utility functions for SVG export.
"""

import svgwrite

def export_to_svg(data, filename="output.svg"):
    print(f"Exporting data to {filename}...")
    dwg = svgwrite.Drawing(filename, profile='tiny')
    for line in data:
        dwg.add(dwg.line(start=line[0], end=line[1], stroke=svgwrite.rgb(10, 10, 16, '%')))
    dwg.save()

def evaluate_quality(data):
    print("Evaluating quality of generated data...")
    # Placeholder for quality evaluation logic
    return "Quality: Good"
