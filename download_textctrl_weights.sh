#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/mnt/p5-modeltests/TextCtrl"
cd "$ROOT"

mkdir -p \
    weights/sd/unet \
    weights/sd/vae \
    weights/sd/scheduler

echo "========== TEXTCTRL-CHECKPOINTS =========="

download_drive() {
    local filename="$1"
    local file_id="$2"
    local dest="weights/$filename"
    local tmp="${dest}.part"

    if [ -s "$dest" ]; then
        echo "Bereits vorhanden: $dest"
        return
    fi

    echo
    echo "Lade herunter: $filename"

    rm -f -- "$tmp"

    uvx --from gdown gdown \
        "https://drive.google.com/uc?id=$file_id" \
        -O "$tmp"

    if [ ! -s "$tmp" ]; then
        echo "FEHLER: Download fehlt oder ist leer: $filename"
        exit 1
    fi

    mv -- "$tmp" "$dest"
    echo "Gespeichert: $dest"
}

download_drive "model.pth" \
    "17JhuVw6-ZK9r858BFwVpN4AiEhNf885g"

download_drive "ocr_model.pth" \
    "1aB_zk7KXzLk1pXMXdSwTebw8mpVSfPc7"

download_drive "style_encoder.pth" \
    "1mBatSbJMjeOeY17GneGP6EwaCP7YnlLo"

download_drive "text_encoder.pth" \
    "1NB0uC6YAy2M5OjJJAnv_JTz4IzWZBBE5"

download_drive "vgg19.pth" \
    "1h94ykc-Bwk0uihb2VHTEXkB5HVNlIIMx"

download_drive "vision_model.pth" \
    "1qRYpLGxlwYcaw1asBPvxL4OqJMK5cGvq"

download_drive "vitstr_base_patch16_224.pth" \
    "1eTbtg5UrFkq3ZWhIv0hRdoLLNLtdB2Zn"

if [ -s /tmp/TextCtrl-Usage-Restriction-Statement ]; then
    cp /tmp/TextCtrl-Usage-Restriction-Statement \
       "weights/Usage Restriction Statement.txt"
fi

echo
echo "========== STABLE DIFFUSION-KONFIGURATIONEN =========="

fetch_small() {
    local url="$1"
    local dest="$2"
    local tmp="${dest}.part"

    if [ -s "$dest" ]; then
        echo "Bereits vorhanden: $dest"
        return
    fi

    echo "Lade herunter: $dest"
    curl --fail --location --retry 5 --retry-delay 2 \
        --output "$tmp" "$url"

    test -s "$tmp"
    mv -- "$tmp" "$dest"
}

BASE="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main"

fetch_small \
    "$BASE/unet/config.json" \
    "weights/sd/unet/config.json"

fetch_small \
    "$BASE/vae/config.json" \
    "weights/sd/vae/config.json"

fetch_small \
    "$BASE/scheduler/scheduler_config.json" \
    "weights/sd/scheduler/scheduler_config.json"

echo
echo "========== FP16-UNET HERUNTERLADEN =========="

download_large() {
    local url="$1"
    local dest="$2"
    local expected_sha="$3"
    local tmp="${dest}.part"
    local actual_sha

    if [ -s "$dest" ]; then
        actual_sha="$(sha256sum "$dest" | cut -d ' ' -f 1)"
        if [ "$actual_sha" = "$expected_sha" ]; then
            echo "Bereits vorhanden und geprüft: $dest"
            return
        fi

        echo "FEHLER: Vorhandene Datei hat falsche Prüfsumme: $dest"
        exit 1
    fi

    echo "Lade herunter: $dest"
    echo "Der Download kann einige Zeit dauern."

    curl --fail --location --retry 5 --retry-delay 3 \
        --continue-at - \
        --output "$tmp" "$url"

    echo "Prüfe SHA256: $dest"
    printf '%s  %s\n' "$expected_sha" "$tmp" | sha256sum --check -

    mv -- "$tmp" "$dest"
    echo "Gespeichert und geprüft: $dest"
}

download_large \
    "$BASE/unet/diffusion_pytorch_model.fp16.safetensors" \
    "weights/sd/unet/diffusion_pytorch_model.safetensors" \
    "c83908253f9a64d08c25fc90874c9c8aef9a329ce1ca5fb909d73b0c83d1ea21"

echo
echo "========== FP16-VAE HERUNTERLADEN =========="

download_large \
    "$BASE/vae/diffusion_pytorch_model.fp16.safetensors" \
    "weights/sd/vae/diffusion_pytorch_model.safetensors" \
    "4fbcf0ebe55a0984f5a5e00d8c4521d52359af7229bb4d81890039d2aa16dd7c"

echo
echo "========== KONFIGURATION ANPASSEN =========="

OLD_PATH="weights/sd/unet/diffusion_pytorch_model.bin"
NEW_PATH="weights/sd/unet/diffusion_pytorch_model.safetensors"

if grep -Fq "$OLD_PATH" configs/inference.yaml; then
    sed -i "s|$OLD_PATH|$NEW_PATH|" configs/inference.yaml
elif grep -Fq "$NEW_PATH" configs/inference.yaml; then
    echo "UNet-Pfad ist bereits korrekt."
else
    echo "FEHLER: UNet-Pfad in configs/inference.yaml nicht gefunden."
    exit 1
fi

echo
echo "========== DATEIEN UND SPEICHER =========="
find weights -type f -printf '%12s Bytes  %p\n' | sort -n
du -sh weights
df -h "$ROOT"

echo
echo "DOWNLOADS ERFOLGREICH ABGESCHLOSSEN"
