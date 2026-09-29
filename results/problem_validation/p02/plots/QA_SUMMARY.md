# P0.2 Figure Quality Assurance

- Figure source validator: strict mode passed, 21 checks passed, 0 warnings, 0 failures (`p02_figure_source_validation.json`).
- Panel alignment: Figures A–C each passed the render-time 1.5 pt alignment and gutter gate.
- PDF text: all three exported PDFs are auditable; smallest detected text is 7 pt, above the 5 pt floor. See the per-figure `*_text_qa.json` files.
- Collision audit: Figures A–C each passed with 0 text/path collisions and 0 warnings. See the per-figure `*_collision_qa.json` files.
- Visual review: all three final PNGs were inspected; axes, angled dataset labels, uncertainty marks, legends, and panel titles are visible without clipping or overlap.
- Exports: PNG preview (300 dpi), TIFF (600 dpi, LZW), editable SVG, and PDF.
