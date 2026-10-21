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

ROOT = Path(__file__).resolve().parent
DATA = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
SEED = 42
STEPS = 50
GUIDANCE = 2.0
STARTING_LAYER = 10

def read_labels(path):
    result = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) == 2:
                result[parts[0]] = parts[1]
    return result

try:
    import os
    os.chdir(ROOT)

    source_dir = DATA / "i_s"
    source_labels = read_labels(DATA / "i_s.txt")
    target_labels = read_labels(DATA / "i_t.txt")
    names = sorted(
        n for n in source_labels
        if n in target_labels and (source_dir / n).is_file()
    )

    if not names:
        raise RuntimeError("Keine gültigen Eingabebilder gefunden.")

    OUT.mkdir(parents=True, exist_ok=False)

    print("Lade TextCtrl-Modell ...", flush=True)
    model = create_model("configs/inference.yaml").cuda()
    state = load_state_dict("weights/model.pth", location="cpu")
    result = model.load_state_dict(state, strict=False)
    del state
    print("Fehlende Schlüssel:", len(result.missing_keys), flush=True)
    print("Unerwartete Schlüssel:", len(result.unexpected_keys), flush=True)
    model.eval()

    monitor_cfg = {
        "max_length": 25,
        "loss_weight": 1.,
        "attention": "position",
        "backbone": "transformer",
        "backbone_ln": 3,
        "checkpoint": "weights/vision_model.pth",
        "charset_path": "src/module/abinet/data/charset_36.txt",
    }
    pipeline = GaMuSA(model, monitor_cfg)

    for idx, name in enumerate(names, 1):
        print(
            f"\n[{idx}/{len(names)}] {name} | "
            f"Ziel={target_labels[name]} | Seed={SEED}",
            flush=True
        )

        with Image.open(source_dir / name) as im:
            original_size = im.size

        source_image = load_image(str(source_dir / name))
        style_image = load_image(str(source_dir / name))

        # Identischer Zufallsstart für jeden Vergleichskandidaten.
        seed_everything(SEED)

        t0 = time.perf_counter()
        result_images = text_editing(
            pipeline,
            source_image,
            style_image,
            source_labels[name],
            target_labels[name],
            starting_layer=STARTING_LAYER,
            ddim_steps=STEPS,
            scale=GUIDANCE,
        )

        generated = result_images[1]
        generated = np.clip(generated * 255, 0, 255).astype(np.uint8)
        result_image = Image.fromarray(generated).resize(
            original_size, Image.Resampling.LANCZOS
        )
        result_image.save(OUT / name)

        del source_image, style_image, result_images
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        print(
            f"Gespeichert: {OUT / name} | "
            f"Dauer: {time.perf_counter() - t0:.1f} s",
            flush=True
        )

    print("\nErstelle Kontaktbogen ...", flush=True)

    # Vergleichssheet, ohne japanische Beschriftungen in der Standardschrift.
    manifest_rows = []
    with (DATA / "benchmark_manifest.tsv").open(
        encoding="utf-8", newline=""
    ) as f:
        import csv
        manifest_rows = list(csv.DictReader(f, delimiter="\t"))

    cols = 3
    cell_w, cell_h = 520, 150
    rows = (len(manifest_rows) + cols - 1) // cols
    sheet = Image.new(
        "RGB", (cols * cell_w, rows * cell_h), (35, 35, 35)
    )
    draw = ImageDraw.Draw(sheet)

    for j, row in enumerate(manifest_rows):
        x = (j % cols) * cell_w
        y = (j // cols) * cell_h

        name = row["input_file"]
        source = Image.open(source_dir / name).convert("RGB")
        generated = Image.open(OUT / name).convert("RGB")

        source.thumbnail((235, 95), Image.Resampling.LANCZOS)
        generated.thumbnail((235, 95), Image.Resampling.LANCZOS)

        draw.text(
            (x + 8, y + 5),
            f"{int(row['test_id']):02d} | candidate {row['candidate_index']} "
            f"| {Path(row['source_image']).stem}",
            fill="white",
        )
        draw.text((x + 8, y + 25), "SOURCE", fill=(200, 200, 200))
        draw.text((x + 270, y + 25), "HELLO OUTPUT", fill=(200, 200, 200))

        sheet.paste(source, (x + 8, y + 45))
        sheet.paste(generated, (x + 270, y + 45))

    comparison = ROOT / "style_benchmark_12_comparison.jpg"
    sheet.save(comparison, quality=95)

    print("\nALLE TESTS ABGESCHLOSSEN.", flush=True)
    print("Ergebnisse:", OUT, flush=True)
    print("Kontaktbogen:", comparison, flush=True)

except Exception:
    print("\nFEHLER IM BENCHMARK:", flush=True)
    traceback.print_exc()
    sys.exit(1)
