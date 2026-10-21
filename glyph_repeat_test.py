#!/usr/bin/env python3
"""Prototype: isolate duplicate glyphs in a TextCtrl result and make a comparison.

This is a CPU-only image-processing test. It does not load TextCtrl or touch source
images. The default sample is style 12 from the completed 2026-10-21 sweep.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_IMAGE = Path(
    "/mnt/p5-modeltests/TextCtrl/style_sweep_20261021_212552/"
    "baseline/style_12_s001_005.png"
)


def log(message: str) -> None:
    print(message, flush=True)


def estimate_background(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    h, w, _ = rgb.shape
    thickness = max(1, min(4, h // 12, w // 12))
    border = np.concatenate(
        [
            rgb[:thickness].reshape(-1, 3),
            rgb[-thickness:].reshape(-1, 3),
            rgb[:, :thickness].reshape(-1, 3),
            rgb[:, -thickness:].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)
    bg = np.median(border, axis=0)
    border_spread = float(np.percentile(np.linalg.norm(border - bg, axis=1), 95))
    return bg, border_spread


def spans_from_activity(active: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(active)
    if len(indices) == 0:
        return []
    spans: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value > previous + 1:
            spans.append((start, previous + 1))
            start = value
        previous = value
    spans.append((start, previous + 1))
    return spans


def find_character_spans(
    distance: np.ndarray, threshold: float, expected_count: int
) -> tuple[list[tuple[int, int]], np.ndarray, float]:
    """Find a contrast threshold that yields exactly one x-span per character.

    Some rendered fonts leave weak antialias/glow pixels connecting two adjacent
    letters in the horizontal projection. Raising the threshold gradually can
    break those weak bridges while keeping the high-contrast glyph cores intact.
    Unlike a blind split, this never cuts through a glyph simply to force a count.
    """
    h, _ = distance.shape
    min_pixels = max(1, int(round(h * 0.008)))
    # Search from the requested threshold upwards and accept the first level that
    # produces the known number of glyph spans. Levels are deterministic and CPU-only.
    candidates = [float(threshold)]
    candidates.extend(float(threshold + step) for step in range(4, 201, 4))
    candidates.extend([220.0, 240.0, 280.0, 320.0, 360.0, 420.0])
    seen: set[float] = set()
    observed: list[tuple[float, int]] = []
    for level in candidates:
        if level in seen:
            continue
        seen.add(level)
        active = (distance >= level).sum(axis=0) >= min_pixels
        spans = spans_from_activity(active)
        observed.append((level, len(spans)))
        if len(spans) == expected_count:
            return spans, (distance >= level), level

    summary = ", ".join(f"{level:g}->{count}" for level, count in observed[:12])
    raise RuntimeError(
        f"Kein sicherer Schwellenwert ergab genau {expected_count} getrennte Zeichenbereiche. "
        f"Erste Versuche (Schwelle->Bereiche): {summary}. "
        "Es wurde bewusst kein beliebiger Schnitt durch einen Buchstaben vorgenommen."
    )


def transparent_glyph(
    rgb: np.ndarray,
    distance: np.ndarray,
    bg: np.ndarray,
    x0: int,
    x1: int,
    threshold: float,
    global_distance_scale: float,
) -> tuple[Image.Image, dict[str, int]]:
    """Extract the pixels in one horizontal character cell as RGBA.

    The output decontaminates the estimated background so this pixel layer can
    be composited over the original clean sprite later. This first prototype is
    intended for near-uniform-background samples; it warns when that is not so.
    """
    h = rgb.shape[0]
    local_d = distance[:, x0:x1]
    low_threshold = max(2.5, threshold * 0.22)
    visible = local_d >= low_threshold
    yy, xx = np.where(visible)
    if len(yy) == 0:
        raise RuntimeError(f"Keine sichtbaren Pixel im Zeichenbereich x={x0}:{x1}.")
    y0 = max(0, int(yy.min()) - 3)
    y1 = min(h, int(yy.max()) + 4)

    crop_rgb = rgb[y0:y1, x0:x1].astype(np.float32)
    delta = crop_rgb - bg.reshape(1, 1, 3)
    mag = np.linalg.norm(delta, axis=2)
    scale = max(float(global_distance_scale), threshold * 1.5, 1.0)
    alpha = np.clip(mag / scale, 0.0, 1.0)
    alpha[mag < low_threshold] = 0.0

    # Recover foreground color from the estimated blend over the background.
    fg = np.broadcast_to(bg.reshape(1, 1, 3), crop_rgb.shape).copy()
    nonzero = alpha > 1e-4
    fg[nonzero] = bg + delta[nonzero] / alpha[nonzero, None]
    fg = np.clip(fg, 0, 255).astype(np.uint8)
    alpha_u8 = np.round(alpha * 255).astype(np.uint8)
    rgba = np.dstack([fg, alpha_u8])

    # Tighten around non-transparent pixels but preserve 2 px horizontal and
    # 3 px vertical room for antialiasing / outer glow.
    ay, ax = np.where(alpha_u8 >= 8)
    if len(ax):
        tx0 = max(0, int(ax.min()) - 2)
        tx1 = min(rgba.shape[1], int(ax.max()) + 3)
        ty0 = max(0, int(ay.min()) - 3)
        ty1 = min(rgba.shape[0], int(ay.max()) + 4)
        rgba = rgba[ty0:ty1, tx0:tx1]
    return Image.fromarray(rgba, "RGBA"), {"y0": y0, "y1": y1, "cell_x0": x0, "cell_x1": x1}


def glyph_core_metrics(
    rgb: np.ndarray,
    distance: np.ndarray,
    span: tuple[int, int],
    threshold: float,
    bg: np.ndarray,
) -> dict[str, float | int]:
    x0, x1 = span
    mask = distance[:, x0:x1] >= threshold
    yy, xx = np.where(mask)
    if len(xx) == 0:
        return {"score": 0.0, "vertical_coverage": 0.0, "foot_coverage": 0.0, "width": 0, "height": 0}
    x_min, x_max = int(xx.min()), int(xx.max()) + 1
    y_min, y_max = int(yy.min()), int(yy.max()) + 1
    core = mask[y_min:y_max, x_min:x_max]
    gh, gw = core.shape
    if gh < 2 or gw < 2:
        return {"score": 0.0, "vertical_coverage": 0.0, "foot_coverage": 0.0, "width": gw, "height": gh}

    # An L is expected to have a long upright stroke and a bottom foot.
    upright_end = max(1, int(gw * 0.58))
    y_start = int(gh * 0.08)
    y_end = max(y_start + 1, int(gh * 0.78))
    upright_slice = core[y_start:y_end, :upright_end]
    vertical_coverage = float(np.mean(upright_slice.sum(axis=1) >= 1)) if upright_slice.size else 0.0

    foot_start = int(gh * 0.70)
    foot_slice = core[foot_start:, :]
    foot_coverage = float(np.mean(foot_slice.sum(axis=0) >= 1)) if foot_slice.size else 0.0

    # Penalise detached speckles lightly. The mask may contain outlines, so this
    # is only a tiny tie-breaker rather than a strong rule.
    ink = int(core.sum())
    density = ink / float(max(1, gh * gw))
    score = 0.98 * (0.68 * vertical_coverage + 0.32 * foot_coverage) + 0.02 * min(density, 1.0)
    return {
        "score": float(score),
        "vertical_coverage": vertical_coverage,
        "foot_coverage": foot_coverage,
        "width": int(gw),
        "height": int(gh),
        "ink_pixels": ink,
    }


def make_checkerboard(size: tuple[int, int], tile: int = 8) -> Image.Image:
    w, h = size
    a = np.empty((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            v = 218 if ((x // tile) + (y // tile)) % 2 == 0 else 164
            a[y, x] = (v, v, v)
    return Image.fromarray(a, "RGB")


def fit_glyph_to_cell(
    source: Image.Image,
    dest_width: int,
    dest_height: int,
    source_core_height: int,
    target_core_height: int,
) -> Image.Image:
    sw, sh = source.size
    desired_scale = target_core_height / max(1, source_core_height)
    max_scale_width = max(0.1, (dest_width - 2) / max(1, sw))
    scale = min(desired_scale, max_scale_width)
    nw = max(1, int(round(sw * scale)))
    nh = max(1, int(round(sh * scale)))
    return source.resize((nw, nh), Image.Resampling.LANCZOS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test automatic replacement of repeated letters in one TextCtrl output.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Existing generated image; no model is run.")
    parser.add_argument("--text", default="HELLO", help="Known generated target text.")
    parser.add_argument("--duplicate", default="L", help="Repeated character to compare and unify.")
    parser.add_argument("--threshold", type=float, default=16.0, help="RGB-distance threshold for core glyph pixels.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory; defaults to a new timestamped sibling folder.")
    args = parser.parse_args()

    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        log(f"FEHLER: Bild nicht gefunden: {image_path}")
        log("Es wurde keine Generierung gestartet und nichts verändert.")
        return 2
    text_value = args.text
    duplicate = args.duplicate
    duplicate_indexes = [i for i, ch in enumerate(text_value) if ch == duplicate]
    if len(duplicate_indexes) != 2:
        log(f"FEHLER: Erwartet werden genau zwei Vorkommen von {duplicate!r} in {text_value!r}; gefunden: {len(duplicate_indexes)}.")
        return 2

    out_dir = args.out_dir
    if out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = image_path.parent.parent / f"glyph_repeat_test_{stamp}"
    out_dir = out_dir.expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        log(f"FEHLER: Ausgabeordner ist nicht leer: {out_dir}")
        log("Bitte ohne Löschen einen neuen --out-dir angeben.")
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    log("[1/5] Bestehendes Bild laden (CPU-only; keine TextCtrl-Inferenz).")
    source = Image.open(image_path)
    if source.mode == "RGBA":
        flat = Image.new("RGBA", source.size, (255, 255, 255, 255))
        flat.alpha_composite(source)
        source = flat.convert("RGB")
    else:
        source = source.convert("RGB")
    rgb = np.asarray(source, dtype=np.uint8)
    h, w, _ = rgb.shape
    bg, border_spread = estimate_background(rgb.astype(np.float32))
    diff = rgb.astype(np.float32) - bg.reshape(1, 1, 3)
    distance = np.linalg.norm(diff, axis=2)
    global_scale = float(np.percentile(distance[distance >= args.threshold], 99.5)) if np.any(distance >= args.threshold) else 1.0
    log(f"    Bildgröße: {w}x{h}; geschätzter Hintergrund RGB={np.round(bg).astype(int).tolist()}; Randabweichung p95={border_spread:.1f}.")
    if border_spread > 12:
        log("    WARNUNG: Der Bildrand ist nicht einheitlich. Der Test kann beim Entfernen des zweiten Buchstabens einen sichtbaren Hintergrundpatch erzeugen.")

    log("[2/5] Zeichenbereiche anhand der Pixelprojektion bestimmen.")
    spans, strong, threshold_used = find_character_spans(distance, args.threshold, len(text_value))
    if len(spans) != len(text_value):
        log(f"FEHLER: {len(spans)} Zeichenbereiche erkannt, aber {len(text_value)} Zeichen erwartet.")
        return 2
    cells: list[tuple[int, int]] = []
    for i in range(len(spans)):
        left = 0 if i == 0 else (spans[i - 1][1] + spans[i][0]) // 2
        right = w if i == len(spans) - 1 else (spans[i][1] + spans[i + 1][0]) // 2
        cells.append((left, right))
    summary = ", ".join(f"{text_value[i]}:{a}-{b}" for i, (a, b) in enumerate(spans))
    log(f"    Erkannte Zeichenpositionen: {summary} (Schwelle={threshold_used:g})")

    log("[3/5] Einzelzeichen mit Transparenz extrahieren und L-Kandidaten bewerten.")
    glyphs: list[Image.Image] = []
    metrics: list[dict[str, float | int]] = []
    crop_meta: list[dict[str, int]] = []
    for i, (x0, x1) in enumerate(cells):
        glyph, meta = transparent_glyph(rgb, distance, bg, x0, x1, args.threshold, global_scale)
        glyphs.append(glyph)
        crop_meta.append(meta)
        metrics.append(glyph_core_metrics(rgb, distance, spans[i], threshold_used, bg))
        log(f"    {i + 1}/{len(text_value)} {text_value[i]}: {glyph.width}x{glyph.height}")
    idx_a, idx_b = duplicate_indexes
    score_a = float(metrics[idx_a]["score"])
    score_b = float(metrics[idx_b]["score"])
    selected_idx = idx_a if score_a >= score_b else idx_b
    rejected_idx = idx_b if selected_idx == idx_a else idx_a
    log(f"    {duplicate}#{idx_a + 1}: score={score_a:.3f}; {duplicate}#{idx_b + 1}: score={score_b:.3f}.")
    log(f"    Automatische Auswahl: {duplicate}#{selected_idx + 1} (heuristische Formbewertung; bitte anhand des Kontaktbildes validieren).")

    log("[4/5] Zweites Vorkommen durch die ausgewählte Glyphe ersetzen.")
    target_x0, target_x1 = cells[rejected_idx]
    target_span = spans[rejected_idx]
    target_rows = np.where(strong[:, target_span[0] : target_span[1]].any(axis=1))[0]
    if len(target_rows) == 0:
        log("FEHLER: Zielbuchstabe besitzt keine erkennbaren Vordergrundpixel.")
        return 2
    target_y0, target_y1 = int(target_rows.min()), int(target_rows.max()) + 1
    source_rows = np.where(strong[:, spans[selected_idx][0] : spans[selected_idx][1]].any(axis=1))[0]
    source_core_height = int(source_rows.max() - source_rows.min() + 1) if len(source_rows) else glyphs[selected_idx].height
    target_core_height = target_y1 - target_y0
    fitted = fit_glyph_to_cell(
        glyphs[selected_idx],
        target_x1 - target_x0,
        h,
        source_core_height,
        target_core_height,
    )
    base = source.copy()
    draw = ImageDraw.Draw(base)
    bg_tuple = tuple(int(round(x)) for x in bg)
    draw.rectangle((target_x0, 0, target_x1 - 1, h - 1), fill=bg_tuple)
    place_x = target_x0 + max(0, ((target_x1 - target_x0) - fitted.width) // 2)
    # Align the replacement's vertical centre with the original glyph's centre.
    place_y = int(round((target_y0 + target_y1 - fitted.height) / 2))
    base_rgba = base.convert("RGBA")
    base_rgba.alpha_composite(fitted, (place_x, place_y))
    unified = base_rgba.convert("RGB")
    unified_path = out_dir / "hello_unified.png"
    unified.save(unified_path)
    selected_path = out_dir / f"selected_{duplicate}.png"
    glyphs[selected_idx].save(selected_path)

    log("[5/5] Vergleichsbilder und Bericht schreiben.")
    scale = 4
    header_h = 28
    gap = 16
    left = source.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
    right = unified.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w * scale * 2 + gap * 3, h * scale + header_h + gap * 2), (238, 238, 238))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((gap, 7), "ORIGINAL", fill=(20, 20, 20), font=font)
    draw.text((w * scale + gap * 2, 7), f"L vereinheitlicht (Quelle: L#{selected_idx + 1})", fill=(20, 20, 20), font=font)
    canvas.paste(left, (gap, header_h + gap))
    canvas.paste(right, (w * scale + gap * 2, header_h + gap))
    comparison_path = out_dir / "comparison_original_vs_unified.png"
    canvas.save(comparison_path)

    tile_w, tile_h = 150, 120
    label_h, tile_gap = 30, 12
    glyph_sheet = Image.new("RGB", (len(glyphs) * tile_w + (len(glyphs) + 1) * tile_gap, tile_h + label_h + 2 * tile_gap), (238, 238, 238))
    gdraw = ImageDraw.Draw(glyph_sheet)
    for i, glyph in enumerate(glyphs):
        x = tile_gap + i * (tile_w + tile_gap)
        label = f"{i + 1}: {text_value[i]}"
        if i in duplicate_indexes:
            label += f" score={float(metrics[i]['score']):.2f}"
        if i == selected_idx:
            label += "  [AUTO]"
        gdraw.text((x, 5), label, fill=(20, 20, 20), font=font)
        board = make_checkerboard((tile_w, tile_h))
        glyph_fit = glyph.copy()
        glyph_fit.thumbnail((tile_w - 12, tile_h - 12), Image.Resampling.LANCZOS)
        board_rgba = board.convert("RGBA")
        board_rgba.alpha_composite(glyph_fit, ((tile_w - glyph_fit.width) // 2, (tile_h - glyph_fit.height) // 2))
        glyph_sheet.paste(board_rgba.convert("RGB"), (x, label_h + tile_gap))
    glyph_sheet_path = out_dir / "isolated_glyphs.png"
    glyph_sheet.save(glyph_sheet_path)

    report = {
        "input_image": str(image_path),
        "text": text_value,
        "duplicate_character": duplicate,
        "image_size": [w, h],
        "estimated_background_rgb": [float(x) for x in bg],
        "border_spread_p95": border_spread,
        "threshold": args.threshold,
        "effective_segmentation_threshold": threshold_used,
        "character_spans": [
            {"index": i + 1, "character": text_value[i], "x_start": int(spans[i][0]), "x_end": int(spans[i][1]),
             "cell_x_start": int(cells[i][0]), "cell_x_end": int(cells[i][1]), "metrics": metrics[i]}
            for i in range(len(text_value))
        ],
        "duplicate_indices_one_based": [idx_a + 1, idx_b + 1],
        "selected_index_one_based": selected_idx + 1,
        "selection_score": {"first": score_a, "second": score_b},
        "note": "Initial geometric L-shape heuristic. Check isolated_glyphs.png and comparison_original_vs_unified.png before considering general use.",
        "outputs": {
            "comparison": str(comparison_path),
            "isolated_glyphs": str(glyph_sheet_path),
            "unified_image": str(unified_path),
            "selected_glyph_rgba": str(selected_path),
        },
    }
    report_path = out_dir / "glyph_repeat_test_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    log("\nFERTIG — keine TextCtrl-Modellläufe durchgeführt.")
    log(f"Original:       {image_path}")
    log(f"Vergleich:      {comparison_path}")
    log(f"Einzelzeichen:  {glyph_sheet_path}")
    log(f"Vereinheitlicht:{unified_path}")
    log(f"Gewähltes L:    {selected_path}")
    log(f"Bericht:        {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"FEHLER: {type(exc).__name__}: {exc}")
        log("Es wurde keine Quellbilddatei überschrieben; prüfen Sie bitte die oben genannte Ursache.")
        raise SystemExit(1)
