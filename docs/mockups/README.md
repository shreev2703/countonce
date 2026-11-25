# UI mockups (approval gate)

Static HTML/CSS pages for PLAN §14.1 and §17. No framework, no build step, no network fonts. Open any page in a browser.
`frontend/` code has **not** been started and stays blocked until Shreev approves these.

| Page | Shows |
|---|---|
| `upload.html` | drag-drop, prompt, optional exemplar, privacy note, §14.5 limits, pre-flight tips |
| `progress.html` | stage list with elapsed time, "processing paused" banner |
| `results.html` | count card, warnings, 3D viewer placeholder, photo strip with legend, correction tools, feedback, downloads, share link |
| `history.html` | past counts |
| `states.html` | empty, loading, and error states (messages copied from `countonce/errors.py`) |

**All numbers are placeholder examples** (47 apples, range 44–51, dates, tokens). Nothing here is a measured result.

Design: one accent color (green); status colors new/duplicate/rejected always come with a text label and a different outline style (solid/dashed/dotted); 16 px+ body text; light/dark via `prefers-color-scheme` plus the "Theme" button; phone layout at 375 px.

Screenshots: `python scripts/render_mockups.py` writes PNGs to `docs/mockups/screenshots/` (needs `playwright` and a browser; otherwise it skips with a message).

## Approval checklist for Shreev

Tick each and reply "approved" (or list changes) in `docs/GATES.md`.

- [ ] Count card is the clear focus of the results page
- [ ] Accent color and overall look are acceptable (light and dark)
- [ ] Upload page: limits, privacy note and pre-flight tips read well
- [ ] Progress: stage names and the "no worker online" banner wording
- [ ] Results: warnings, 3D viewer panel, numbered markers and two-way highlighting idea
- [ ] Overlay legend (NEW / DUPLICATE / REJECTED) is understandable without color
- [ ] Correction affordances (split / merge / reject) are discoverable
- [ ] Feedback, downloads, share link and revoke/delete placement
- [ ] History table, and its phone layout
- [ ] Error messages are friendly and actionable
- [ ] Phone layout (375 px) is usable one-handed
- [ ] Keyboard: Tab order and focus outlines look right
