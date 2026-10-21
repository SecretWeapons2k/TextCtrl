import csv
import gc
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from pytorch_lightning import seed_everything

from inference import load_image
from utils import create_model, load_state_dict
from src.MuSA.GaMuSA import GaMuSA
from src.MuSA.GaMuSA_app import text_editing


SEED = 42
DEFAULT_GUIDANCE = 2.0
DEFAULT_START_STEP = 24
DEFAULT_START_LAYER = 10
DEFAULT_STEPS = 50

SELECTED_TEST_IDS = {1, 3, 4, 5, 8, 10}


def read_manifest(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    rows.sort(key=lambda r: int(r["test_id"]))

    if len(rows) != 12:
        raise RuntimeError(
            f"Erwartet wurden 12 Baseline-Fälle, gefunden: {len(rows)}"
        )

    if len({r["input_file"] for r in rows}) != len(rows):
        raise RuntimeError("Doppelte Eingabedateien im Benchmark-Manifest.")

    return rows


def value_key(value):
    if isinstance(value, float):
        return f"{value:g}".replace(".", "p")
    return str(value)


def build_runs(rows):
    selected = [
        row for row in rows
        if int(row["test_id"]) in SELECTED_TEST_IDS
    ]

    if len(selected) != 6:
        raise RuntimeError(
            f"Die sechs ausgewählten Testfälle fehlen oder sind doppelt: "
            f"{len(selected)} gefunden."
        )

    runs = []

    for value in (1.5, 2.5, 3.0):
        for row in rows:
            runs.append({
                "phase": "guidance_scale",
                "parameter": "guidance_scale",
                "value": value,
                "row": row,
                "guidance_scale": value,
                "start_step": DEFAULT_START_STEP,
                "starting_layer": DEFAULT_START_LAYER,
                "num_inference_steps": DEFAULT_STEPS,
            })

    for value in (16, 20, 28, 32):
        for row in selected:
            runs.append({
                "phase": "start_step",
                "parameter": "start_step",
                "value": value,
                "row": row,
                "guidance_scale": DEFAULT_GUIDANCE,
                "start_step": value,
                "starting_layer": DEFAULT_START_LAYER,
                "num_inference_steps": DEFAULT_STEPS,
            })

    for value in (6, 8, 12, 14):
        for row in selected:
            runs.append({
                "phase": "starting_layer",
                "parameter": "starting_layer",
                "value": value,
                "row": row,
                "guidance_scale": DEFAULT_GUIDANCE,
                "start_step": DEFAULT_START_STEP,
                "starting_layer": value,
                "num_inference_steps": DEFAULT_STEPS,
            })

    for value in (30, 40, 60, 75):
        for row in selected:
            runs.append({
                "phase": "num_inference_steps",
                "parameter": "num_inference_steps",
                "value": value,
                "row": row,
                "guidance_scale": DEFAULT_GUIDANCE,
                "start_step": DEFAULT_START_STEP,
                "starting_layer": DEFAULT_START_LAYER,
                "num_inference_steps": value,
            })

    if len(runs) != 108:
        raise RuntimeError(f"Interner Fehler: {len(runs)} statt 108 Läufe.")

    return runs, selected


def make_monitor_config():
    return {
        "max_length": 25,
        "loss_weight": 1.0,
        "attention": "position",
        "backbone": "transformer",
        "backbone_ln": 3,
        "checkpoint": "weights/vision_model.pth",
        "charset_path": "src/module/abinet/data/charset_36.txt",
    }


def generate_one(model_pipeline, row, config, data_dir, output_path):
    input_path = data_dir / "i_s" / row["input_file"]

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    with Image.open(input_path) as im:
        original_size = im.size

    # Laden liefert dieselbe 256x256-Vorbereitung wie die offizielle Inferenz.
    source_image = load_image(str(input_path))
    style_image = load_image(str(input_path))

    # Jeder Kandidat erhält denselben Zufallsstart für faire Vergleiche.
    seed_everything(SEED)

    result_images = text_editing(
        model_pipeline,
        source_image,
        style_image,
        row["source_text"],
        row["target_text"],
        starting_layer=int(config["starting_layer"]),
        start_step=int(config["start_step"]),
        ddim_steps=int(config["num_inference_steps"]),
        scale=float(config["guidance_scale"]),
    )

    generated = np.clip(
        result_images[1] * 255.0, 0, 255
    ).astype(np.uint8)

    result_image = Image.fromarray(generated).resize(
        original_size, Image.Resampling.LANCZOS
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_image.save(output_path)

    del source_image, style_image, result_images, result_image
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_rgb(path):
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:
        return None


def make_contact_sheet(
    root,
    data_dir,
    rows,
    phase,
    parameter_values,
):
    labels = ["SOURCE", "BASELINE"] + [
        value_key(v) for v in parameter_values
    ]

    label_w = 112
    cell_w = 180
    header_h = 38
    row_h = 110
    sheet = Image.new(
        "RGB",
        (
            label_w + cell_w * len(labels),
            header_h + row_h * len(rows),
        ),
        (35, 35, 35),
    )
    draw = ImageDraw.Draw(sheet)

    for col, label in enumerate(labels):
        x = label_w + col * cell_w
        draw.text((x + 5, 12), label, fill=(245, 245, 245))

    for row_index, row in enumerate(rows):
        y = header_h + row_index * row_h
        test_label = (
            f"{int(row['test_id']):02d} "
            f"{Path(row['source_image']).stem}"
        )
        draw.text((5, y + 8), test_label[:18], fill=(245, 245, 245))

        name = row["input_file"]
        source_path = data_dir / "i_s" / name
        baseline_path = root / "baseline" / name

        paths = [source_path, baseline_path]
        for value in parameter_values:
            paths.append(
                root / "results" / phase / value_key(value) / name
            )

        for col, path in enumerate(paths):
            image = load_rgb(path)
            if image is None:
                continue

            image.thumbnail((cell_w - 10, row_h - 28), Image.Resampling.LANCZOS)
            x = label_w + col * cell_w
            px = x + (cell_w - image.width) // 2
            py = y + 25 + (row_h - 28 - image.height) // 2
            sheet.paste(image, (px, py))

    output = root / f"contact_{phase}.jpg"
    sheet.save(output, quality=94)
    print(f"[SHEET] {output}", flush=True)


def main():
    if len(sys.argv) < 5:
        print(
            "Aufruf: script.py ROOT DATA BASELINE OUT [--plan]",
            file=sys.stderr,
        )
        return 2

    root = Path(sys.argv[1]).resolve()
    data_dir = Path(sys.argv[2]).resolve()
    baseline_dir = Path(sys.argv[3]).resolve()
    output_dir = Path(sys.argv[4]).resolve()
    plan_only = "--plan" in sys.argv[5:]

    import os
    os.chdir(root)

    manifest_path = data_dir / "benchmark_manifest.tsv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    rows = read_manifest(manifest_path)
    runs, selected = build_runs(rows)

    if output_dir.exists() and not plan_only:
        raise FileExistsError(
            f"Ausgabeordner existiert bereits: {output_dir}"
        )

    missing = []
    for row in rows:
        if not (data_dir / "i_s" / row["input_file"]).is_file():
            missing.append(str(data_dir / "i_s" / row["input_file"]))
        if not (baseline_dir / row["input_file"]).is_file():
            missing.append(str(baseline_dir / row["input_file"]))

    if missing:
        raise RuntimeError(
            "Eingaben oder Baseline fehlen:\n" + "\n".join(missing[:30])
        )

    print("========== SWEEP-PLAN ==========", flush=True)
    print("Baseline-Fälle vorhanden:", len(rows), flush=True)
    print("Neue Generierungen:", len(runs), flush=True)
    print("guidance_scale: 36", flush=True)
    print("start_step: 24", flush=True)
    print("starting_layer: 24", flush=True)
    print("num_inference_steps: 24", flush=True)
    print("Seed:", SEED, flush=True)
    print("Ausgabe:", output_dir, flush=True)

    if plan_only:
        print("PLAN-PRÜFUNG ERFOLGREICH; es wurde nichts generiert.", flush=True)
        return 0

    output_dir.mkdir(parents=True, exist_ok=False)
    baseline_out = output_dir / "baseline"
    baseline_out.mkdir()

    for row in rows:
        shutil.copy2(
            baseline_dir / row["input_file"],
            baseline_out / row["input_file"],
        )

    results_path = output_dir / "sweep_results.tsv"
    error_path = output_dir / "sweep_errors.log"

    columns = [
        "run_id", "phase", "parameter", "value", "test_id",
        "candidate_index", "source_image", "source_text",
        "target_text", "seed", "guidance_scale", "start_step",
        "starting_layer", "num_inference_steps", "status",
        "duration_seconds", "output_file", "error",
    ]

    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, delimiter="\t"
        )
        writer.writeheader()

        # Vorhandene Baseline dokumentieren, nicht erneut generieren.
        for row in rows:
            writer.writerow({
                "run_id": "BASE",
                "phase": "baseline_existing",
                "parameter": "baseline",
                "value": "2.0/24/10/50",
                "test_id": row["test_id"],
                "candidate_index": row["candidate_index"],
                "source_image": row["source_image"],
                "source_text": row["source_text"],
                "target_text": row["target_text"],
                "seed": SEED,
                "guidance_scale": DEFAULT_GUIDANCE,
                "start_step": DEFAULT_START_STEP,
                "starting_layer": DEFAULT_START_LAYER,
                "num_inference_steps": DEFAULT_STEPS,
                "status": "baseline_existing",
                "duration_seconds": "",
                "output_file": str(
                    baseline_out / row["input_file"]
                ),
                "error": "",
            })
        handle.flush()

        print("\n========== MODELL LADEN ==========", flush=True)
        model = create_model("configs/inference.yaml").cuda()
        checkpoint = load_state_dict("weights/model.pth", location="cpu")
        load_result = model.load_state_dict(checkpoint, strict=False)
        del checkpoint

        print(
            "Checkpoint geladen; fehlende Schlüssel:",
            len(load_result.missing_keys),
            "| unerwartete Schlüssel:",
            len(load_result.unexpected_keys),
            flush=True,
        )
        model.eval()

        pipeline = GaMuSA(model, make_monitor_config())
        failures = 0
        completed = 0

        for number, run in enumerate(runs, 1):
            row = run["row"]
            config = {
                "guidance_scale": run["guidance_scale"],
                "start_step": run["start_step"],
                "starting_layer": run["starting_layer"],
                "num_inference_steps": run["num_inference_steps"],
            }

            name = row["input_file"]
            phase_dir = (
                output_dir / "results" / run["phase"]
                / value_key(run["value"])
            )
            output_path = phase_dir / name

            print(
                f"\n[SWEEP {number:03d}/{len(runs):03d}] "
                f"{run['phase']}={run['value']} | "
                f"Test {row['test_id']}/12: {row['source_image']}",
                flush=True,
            )
            print(
                "  Parameter: "
                f"guidance={config['guidance_scale']} "
                f"start_step={config['start_step']} "
                f"start_layer={config['starting_layer']} "
                f"steps={config['num_inference_steps']}",
                flush=True,
            )

            started = time.perf_counter()
            status = "ok"
            error_text = ""

            try:
                generate_one(
                    pipeline, row, config, data_dir, output_path
                )
                completed += 1
                print(
                    f"[SWEEP RESULT] OK | {completed}/{len(runs)} "
                    f"erfolgreich | {time.perf_counter()-started:.1f} s "
                    f"| {output_path}",
                    flush=True,
                )
            except Exception as exc:
                failures += 1
                status = "error"
                error_text = (
                    f"{type(exc).__name__}: {exc}"
                ).replace("\n", " ")
                with error_path.open("a", encoding="utf-8") as err:
                    err.write(
                        f"\n[SWEEP {number:03d}] {run['phase']}="
                        f"{run['value']} | {name}\n"
                    )
                    err.write(traceback.format_exc())
                print(
                    f"[SWEEP RESULT] FEHLER | {error_text}",
                    flush=True,
                )
                traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            writer.writerow({
                "run_id": f"{number:03d}",
                "phase": run["phase"],
                "parameter": run["parameter"],
                "value": run["value"],
                "test_id": row["test_id"],
                "candidate_index": row["candidate_index"],
                "source_image": row["source_image"],
                "source_text": row["source_text"],
                "target_text": row["target_text"],
                "seed": SEED,
                "guidance_scale": config["guidance_scale"],
                "start_step": config["start_step"],
                "starting_layer": config["starting_layer"],
                "num_inference_steps": config["num_inference_steps"],
                "status": status,
                "duration_seconds": round(time.perf_counter()-started, 3),
                "output_file": str(output_path) if status == "ok" else "",
                "error": error_text,
            })
            handle.flush()

        print("\n========== KONTAKTBOGEN ERSTELLEN ==========", flush=True)
        make_contact_sheet(
            output_dir, data_dir, rows,
            "guidance_scale", (1.5, 2.5, 3.0)
        )
        make_contact_sheet(
            output_dir, data_dir, selected,
            "start_step", (16, 20, 28, 32)
        )
        make_contact_sheet(
            output_dir, data_dir, selected,
            "starting_layer", (6, 8, 12, 14)
        )
        make_contact_sheet(
            output_dir, data_dir, selected,
            "num_inference_steps", (30, 40, 60, 75)
        )

        print("\n========== SWEEP ABGESCHLOSSEN ==========", flush=True)
        print("Erfolgreiche Generierungen:", completed, "/", len(runs), flush=True)
        print("Fehler:", failures, flush=True)
        print("Ergebnisse:", output_dir, flush=True)
        print("Manifest:", results_path, flush=True)
        print("Kontaktbögen: contact_*.jpg", flush=True)

        return 0 if failures == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("\nFATALER SWEEP-FEHLER:", flush=True)
        traceback.print_exc()
        raise
