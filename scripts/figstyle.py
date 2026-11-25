#!/usr/bin/env python3
"""Shared typography and final print size for every paper panel.

The font and the way panels are saved are kept in one place so the generator
scripts cannot drift.

``bbox_inches="tight"`` trims the page to the ink, which makes a panel's final
dimensions a function of how long its tick labels happen to be. Panels saved
that way cannot be assembled onto a fixed grid, and scaling them afterwards is
not a remedy: placed-PDF text scales with the artwork, so an 80% scale turns a
5.2 pt annotation into 4.2 pt and breaks the 5 pt floor silently.

``fit_save`` inverts the relationship. State the panel's final size in
millimetres and it adjusts ``figsize`` until the tight bounding box is that
size, then pads the saved bbox to the target exactly. Font sizes are in points
and do not move when ``figsize`` does, so the knob turns plot area against
label area and never touches type size. The PDF MediaBox then equals the size
asked for, and the panel is placed at 100%. If the target is too small for the
content the fit cannot converge, and the helper says so by name and millimetre
instead of clipping.

The specification it serves (nature.com):

* "Nature's standard figure sizes are 89 mm (single column) and 183 mm
  (double column)";  "The maximum height for a Nature figure is 170 mm".
* "All text should be a sans-serif typeface, preferably Helvetica or Arial."
* "Minimum text size: 5pt." / "Maximum text size for all other text: 7pt."
* "Separate panels ... should be labelled with 8 pt bold, upright ... a, b, c's"
* "Embed fonts (True Type 2 or 42), do not select TrueType 3."

The face is Liberation Sans. It is Arial-metric, Arial is one of the two faces
the specification names, and it is a real TrueType file that embeds as
``CID TrueType`` under ``pdf.fonttype: 42``. A CFF-flavoured OpenType face
embeds its CFF table into a descriptor that claims TrueType, which is the
mismatch the "True Type 2 or 42" rule exists to exclude.
"""
from pathlib import Path

import matplotlib
from matplotlib.transforms import Bbox

# The one place the face is named. See the module docstring for why.
FONT = "Liberation Sans"

MM = 1.0 / 25.4          # mm -> inches
COL1_MM, COL2_MM = 89.0, 183.0
MAX_H_MM = 170.0
COL1, COL2 = COL1_MM * MM, COL2_MM * MM

# Type sizes, all inside Nature's 5-7 pt window. PANEL_PT is the single
# documented exception to the 7 pt ceiling.
BASE_PT = 7.0            # body text, at the ceiling
TITLE_PT = 7.0           # panel titles, at the ceiling
SMALL_PT = 6.5
TINY_PT = 5.5            # smallest type used anywhere; floor is 5.0
PANEL_PT = 8.0           # "8 pt bold, upright"

#: rcParams every panel shares. Scripts may override, but not the type sizes.
BASE_RC = {
    "font.family": FONT,
    "font.size": BASE_PT,
    "axes.titlesize": TITLE_PT,
    "axes.labelsize": BASE_PT,
    "xtick.labelsize": BASE_PT,
    "ytick.labelsize": BASE_PT,
    "legend.fontsize": SMALL_PT,
    # Type 42 = TrueType: text stays SELECTABLE when the pdf is placed in
    # Illustrator. matplotlib's default Type 3 imports as outline soup, which
    # violates Nature's "Do not outline text" with no way back.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    # Mathtext ($\kappa$, $\rho$, $k$) otherwise renders from matplotlib's
    # bundled DejaVu, so a panel with one Greek letter ships TWO typefaces and
    # breaks "all text ... one sans-serif typeface". Point it at our face.
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT,
    "mathtext.it": FONT + ":italic",
    "mathtext.bf": FONT + ":bold",
    # The remaining mathtext slots have to be named too. With fontset "custom"
    # matplotlib resolves ALL of them when it builds the math font table, and the
    # unset ones default to families this machine does not have ("cursive" ->
    # "findfont: Font family ['cursive'] not found. Falling back to DejaVu Sans").
    # Nothing in these figures uses \mathcal, but a silent DejaVu fallback in
    # a font table admits a second typeface, and Nature asks for one sans-serif
    # face throughout.
    "mathtext.sf": FONT,
    "mathtext.tt": FONT,
    "mathtext.cal": FONT + ":italic",
    # Measure text with the font's own (unhinted) metrics. The PDF backend
    # typesets from those metrics, but the Agg layout pass hints to whole
    # pixels, so with hinting on a string measures ~1% narrower at the layout
    # dpi than it finally sets -- enough to push the last glyph of an
    # overhanging annotation off a page sized from that measurement.
    "text.hinting": "none",
}


def _tight(fig):
    fig.canvas.draw()
    return fig.get_tightbbox(fig.canvas.get_renderer())


def sub_letter(ax, letter, pad_mm=1.2):
    """A panel letter for ONE AXES of a multi-panel row.

    `panel_letter` anchors in FIGURE fraction at the page's top-left corner, so
    it can letter one panel per file and no more. A file that holds several
    LETTERED panels side by side -- Figure 2's d, e and f -- needs each letter
    pinned to its own axes instead.

    The distance is measured rather than fixed. What sits to the left of an
    axes is its tick labels and its y-label, and those are not the same width
    twice, so an offset in points puts the letter a different distance from the
    ink on every panel. The letter is placed `pad_mm` to the left of the axes'
    own tight bounding box, the same rule `panel_letter` uses for the page.
    """
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    fw = fig.get_size_inches()[0]
    t = fig.text(bb.x0 - (pad_mm * MM / 72.0) / fw, bb.y1, letter,
                 fontsize=PANEL_PT, fontweight="bold", ha="right", va="bottom")
    return t


def fit_save(fig, out_dir, name, mm, dpi=600, exts=("pdf", "png"), tol=0.004):
    """Save `fig` so its page is EXACTLY `mm` = (width_mm, height_mm).

    Rescales ``figsize`` until the tight bbox matches the target, then writes
    with an explicit bbox padded to the target, so the PDF MediaBox is the
    requested size to within rounding. Returns the achieved (w_mm, h_mm).

    Font sizes are in points and are invariant under this rescaling: the only
    thing that moves is how much of the page the axes occupy. That is the
    property that makes "regenerate at final size" safe where "scale in
    Illustrator" is not.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    W, H = mm[0] * MM, mm[1] * MM

    # Fixed-point iteration. Margins (tick labels, titles) are roughly additive
    # constants in inches, so the map figsize -> tightbbox is close to affine
    # with slope 1 and converges in a handful of steps.
    bb = _tight(fig)
    for _ in range(14):
        fw, fh = fig.get_size_inches()
        sw, sh = W / bb.width, H / bb.height
        if abs(sw - 1) < tol and abs(sh - 1) < tol:
            break
        nw, nh = max(fw * sw, 0.3), max(fh * sh, 0.3)
        fig.set_size_inches(nw, nh)
        bb = _tight(fig)

    # Whether the layout can hold the slot is decided here, on the converged
    # residual, before any further nudging.
    got = (bb.width / MM, bb.height / MM)
    bad = abs(got[0] - mm[0]) > 0.6 or abs(got[1] - mm[1]) > 0.6
    tgt = (mm[0], mm[1])

    if not bad:
        # Approach the target from BELOW. The padded box is the target exactly,
        # so a tight bbox even a fraction over it would be clipped -- and what
        # gets clipped is the outermost annotation, the thing you least want
        # silently trimmed. (This bit the 183x40 mm Task II violin: a 0.4 mm
        # overshoot ate the last letter of "5x5 ensemble median".)
        # 0.4 mm of slack per side. The value follows from get_tightbbox, which returns
        # the axes boundary WITHOUT the spine's stroke, so half of a 0.6 pt
        # spine (0.3 pt = 0.11 mm), plus antialiasing, sits outside the measured
        # box. Measured on the 2026-10-24 drop before this line existed: 24 of
        # 35 panels had the bottom spine's lower half shaved off by the right
        # page edge. (matplotlib's own bbox_inches="tight" has the same flaw --
        # it is why the 10/20 panels also carried ink on the boundary.)
        safe_w, safe_h = W - 0.8 * MM, H - 0.8 * MM
        for _ in range(8):
            if bb.width <= safe_w and bb.height <= safe_h:
                break
            fig.set_size_inches(*(s * min(1.0, t / c)
                                  for s, t, c in zip(fig.get_size_inches(),
                                                     (safe_w, safe_h),
                                                     (bb.width, bb.height))))
            bb = _tight(fig)

    if bad:
        # Do not clip. Say so and keep the ink. A panel that will not shrink to
        # the slot needs its LAYOUT changed and its page left untrimmed.
        print(f"  !! {name}: wanted {tgt[0]:.0f}x{tgt[1]:.0f} mm, "
              f"settled at {got[0]:.1f}x{got[1]:.1f} mm -- layout is too dense")
        box = bb
    else:
        # Centre the saved page on the measured ink.
        H_out = H
        cx, cy = 0.5 * (bb.x0 + bb.x1), 0.5 * (bb.y0 + bb.y1)
        box = Bbox.from_extents(cx - W / 2, cy - H_out / 2,
                                cx + W / 2, cy + H_out / 2)

    import matplotlib.pyplot as plt
    for ext in exts:
        fig.savefig(out_dir / f"{name}.{ext}", dpi=dpi, bbox_inches=box)
    plt.close(fig)
    w_mm, h_mm = box.width / MM, box.height / MM
    flag = ""
    if w_mm > COL2_MM + 0.5:
        flag = "  <-- OVER 183 mm"
    elif h_mm > MAX_H_MM + 0.5:
        flag = "  <-- OVER 170 mm"
    print(f"  {name}.pdf / .png   {w_mm:6.1f} x {h_mm:6.1f} mm{flag}")
    return w_mm, h_mm


def panel_letter(ax, letter):
    """Draw `letter` at the PANEL PAGE's top-left corner, 8 pt bold upright.

    Nature states the type ("Separate panels in multi-part figures should be
    labelled with 8 pt bold, upright (not italic) a, b, c's") and says nothing
    about position. Position still matters, because a letter whose distance
    from the page edge changes panel by panel is exactly what a reader notices
    once the panels are stacked into one figure.

    The anchor is a FIGURE fraction, not an axes fraction. An axes-fraction
    indent scales with the panel, so panels at several widths in one figure put
    their letters on different vertical lines. Pinning each letter to its own
    page corner makes them land flush left and flush top at any panel width
    once ``fit_save`` crops to the tight bbox.
    """
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = fig.get_tightbbox(r)                       # inches, ink only
    W, H = fig.get_size_inches()
    t = fig.text(0.0, 0.0, letter, fontsize=PANEL_PT, fontweight="bold",
                 va="top", ha="left")
    # Place it just OUTSIDE the ink on the left and level with the top of it.
    # Pinning it to a fixed figure fraction instead (x = 0) looks right until you
    # measure the result: matplotlib's default left margin is 0.125 of the figure
    # and a rotated y-label only reaches back into part of it, so a letter at
    # x = 0 became the leftmost ink and dragged ~16 mm of empty page into the
    # saved bbox -- 9% of a 180 mm panel, against Nature BE's "minimizing white
    # space". Measuring the ink first costs one extra draw and 2-3 mm.
    tw = t.get_window_extent(r).width / fig.dpi     # the letter's own width
    t.set_position(((bb.x0 - tw - 3.0 / 72.0) / W, bb.y1 / H))
    return t


def check_font():
    """Fail loudly rather than silently falling back to DejaVu Sans."""
    from matplotlib import font_manager as fm
    try:
        path = fm.findfont(FONT, fallback_to_default=False)
    except Exception:                                    # pragma: no cover
        raise SystemExit(f"figstyle: font {FONT!r} is not installed; refusing "
                         f"to render with a substitute of different metrics")
    if not path.lower().endswith((".ttf", ".ttc")):
        print(f"  WARNING: {FONT} resolved to {path} -- not a TrueType file, "
              f"so pdf.fonttype=42 may embed a mismatched font program")
    return path
