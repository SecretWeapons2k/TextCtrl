#!/usr/bin/env bash

ROOT="/mnt/p5-modeltests/TextCtrl"
DEST="$ROOT/weights/sd/vae/diffusion_pytorch_model.bin"
TMP="$DEST.part"
URL="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/vae/diffusion_pytorch_model.bin"
SHA="1b134cded8eb78b184aefb8805b6b572f36fa77b255c483665dda931fa0130c5"

cd "$ROOT" || exit 1

if [ -f "$DEST" ]; then
    if printf '%s  %s\n' "$SHA" "$DEST" | sha256sum --check -; then
        echo "VAE-BIN ist bereits vorhanden und gültig."
        exit 0
    else
        echo "FEHLER: Vorhandene VAE-BIN-Datei hat eine falsche Prüfsumme."
        exit 1
    fi
fi

echo "Lade VAE im kompatiblen BIN-Format herunter ..."
if ! curl --fail --location --retry 5 --retry-delay 3 \
    --continue-at - --output "$TMP" "$URL"; then
    echo "FEHLER beim Download. Eine Teildatei bleibt für einen Wiederholungsversuch erhalten."
    exit 1
fi

echo
echo "Prüfe SHA-256 ..."
if ! printf '%s  %s\n' "$SHA" "$TMP" | sha256sum --check -; then
    echo "FEHLER: Prüfsumme stimmt nicht. Die Zieldatei wird nicht installiert."
    exit 1
fi

if ! mv -- "$TMP" "$DEST"; then
    echo "FEHLER beim Verschieben der geprüften Datei."
    exit 1
fi

echo
echo "VAE-BIN erfolgreich heruntergeladen und geprüft."
ls -lh "$DEST"
