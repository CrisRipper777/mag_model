# P0.1+ Figure QA Notes

## Figure contract

- **Figure-level claim:** frozen-probe similarity stratifies the sign of the existing CE message utility, but low-sim beneficial and high-sim harmful rates must be compared with the overall base rate to determine enrichment.
- **Archetype:** quantitative two-panel grid.
- **Panel a:** Text utility sign rates by probe-similarity quintile.
- **Panel b:** Visual utility sign rates by probe-similarity quintile.
- **Statistics:** each point is the mean of three independent seeds; the band is ± population SD across seeds. Dashed lines are the overall beneficial/harmful sign base rates. The per-seed CSV carries the separate target-node bootstrap intervals.
- **Population:** fixed sampled validation relations from the existing P0.1 artifacts; no graph-wide edge claim is made.

## Automated checks

- All five render-time panel-alignment reports: `PASS` at 1.5 pt tolerance; no warnings or exemptions.
- All five PDF text audits: minimum font run 6.7 pt; no text below the 5 pt floor.
- All five rendered PDF collision audits: `PASS`; zero failures, zero warnings, and no contained overlays.
- Python source preflight: 19 passes, 0 failures, 2 warnings. The warnings are intentional for this report bundle: PNG is a 300-dpi preview and no TIFF is supplied; editable vector PDF/SVG are the figure masters, and no journal submission target was specified.
- Visual review covered both panels in all five datasets. Panel titles/labels, common axes, uncertainty bands, overall base-rate lines, legend mapping, and final-size text are legible. ele-fashion Visual retains its wider seed-SD band; Reddit-S Visual retains its non-monotone profile rather than smoothing or omitting it.

The exact machine-readable alignment, PDF-text, collision, and source-preflight reports are stored beside the figures or at the result-bundle root.
