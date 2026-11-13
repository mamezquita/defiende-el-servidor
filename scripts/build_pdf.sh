#!/usr/bin/env bash
# Convierte informe/informe.html en informe/informe.pdf con Chrome sin ventana, y comprueba que el
# cuerpo, sin la página de referencias, no pase de 5 páginas.
#
#   python -m dsrv.build_report && bash scripts/build_pdf.sh
#
# En WSL no hay pandoc ni LaTeX: usa el Chrome (o Edge) de Windows con un perfil temporal, que no
# toca la sesión del navegador. En Linux usa chromium o google-chrome si están instalados.

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
HTML="$ROOT/informe/informe.html"
PDF="$ROOT/informe/informe.pdf"
MAX_BODY_PAGES=5
REFERENCE_PAGES=1

[ -f "$HTML" ] || { echo "ERROR: falta informe/informe.html; ejecuta python -m dsrv.build_report" >&2; exit 2; }

BROWSER=""
for candidate in \
    "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" \
    "/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe" \
    "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" \
    "$(command -v chromium || true)" "$(command -v google-chrome || true)"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then BROWSER=$candidate; break; fi
done
[ -n "$BROWSER" ] || { echo "ERROR: no hay Chrome, Edge ni chromium. Instala chromium o pandoc." >&2; exit 2; }

if [[ "$BROWSER" == /mnt/* ]]; then
    # Chrome de Windows: trabaja en una carpeta temporal de Windows.
    win_temp=$(cd /mnt/c && powershell.exe -NoProfile -Command '[IO.Path]::GetTempPath()' | tr -d '\r')
    work=$(wslpath "$win_temp")/informe-pdf-$$
    mkdir -p "$work"
    cp "$HTML" "$work/informe.html"
    win_work=$(wslpath -w "$work")
    (cd /mnt/c && "$BROWSER" --headless=new --disable-gpu --no-first-run --no-default-browser-check \
        --user-data-dir="$win_work\\perfil" --no-pdf-header-footer \
        --print-to-pdf="$win_work\\informe.pdf" "file:///$(wslpath -m "$work/informe.html")" \
        >/dev/null 2>&1)
    cp "$work/informe.pdf" "$PDF"
    rm -rf "$work"
else
    work=$(mktemp -d)
    "$BROWSER" --headless=new --disable-gpu --user-data-dir="$work/perfil" --no-pdf-header-footer \
        --print-to-pdf="$PDF" "file://$HTML" >/dev/null 2>&1
    rm -rf "$work"
fi

pages=$(python3 -c "import re,sys; print(len(re.findall(rb'/Type\s*/Page[^s]', open(sys.argv[1],'rb').read())))" "$PDF")
body=$((pages - REFERENCE_PAGES))
echo "informe/informe.pdf: $pages páginas, $body sin referencias (máximo $MAX_BODY_PAGES)"
[ "$body" -le "$MAX_BODY_PAGES" ] || { echo "ERROR: el cuerpo pasa de $MAX_BODY_PAGES páginas" >&2; exit 1; }
