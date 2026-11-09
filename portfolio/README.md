# Project portfolio

[ModelPicker-Portfolio.pdf](ModelPicker-Portfolio.pdf) contains one page of project capabilities and connection instructions followed by three pages of the recorded Claude Desktop session.

The conversation images are preserved in `session-assets/`; the two connection screenshots are included alongside them. Tool names were checked against `modelpicker/server.py`, and the recorded comparison was checked against the owner's saved evaluation before publication. Runtime data and credentials are not included.

To rebuild the PDF, use a separate environment with `reportlab` installed, DejaVu Sans/DejaVu Sans Mono fonts under `/usr/share/fonts/truetype/dejavu/`, and run:

```bash
python portfolio/build_final_portfolio.py
```

These PDF-generation dependencies are separate from the ModelPicker inference runtime.
