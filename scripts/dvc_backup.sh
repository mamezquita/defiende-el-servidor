#!/usr/bin/env bash
# Respaldo del remote DVC mientras DagsHub no está disponible.
#
#   bash scripts/dvc_backup.sh crear    remote local "respaldo", dvc push y ZIP con LEEME.txt
#   bash scripts/dvc_backup.sh probar   clon nuevo desde GitHub, dvc pull desde el ZIP y huellas
#
# Carpeta base: $MLOPS_DIR, por defecto ~/mlops. El remote por defecto (origin) no cambia: "respaldo"
# se registra con --local, en .dvc/config.local, que Git ignora.

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BASE=${MLOPS_DIR:-"$HOME/mlops"}
REMOTE_DIR="$BASE/dvc-respaldo"
ZIP="$BASE/entrega/dvc-remote-respaldo.zip"
REPO_URL="https://github.com/mamezquita/defiende-el-servidor.git"
DVC="$ROOT/.venv/bin/dvc"
PY="$ROOT/.venv/bin/python"
EVIDENCE="$ROOT/evidencias/respaldo-dvc.json"

crear() {
    cd "$ROOT"
    "$DVC" remote add --local --force respaldo "$REMOTE_DIR"
    "$DVC" push -r respaldo
    "$DVC" status -c -r respaldo
    mkdir -p "$(dirname "$ZIP")"
    rm -f "$ZIP"
    "$PY" - "$REMOTE_DIR" "$ROOT/scripts/LEEME-respaldo.txt" "$ZIP" <<'EOF'
import sys, zipfile
from pathlib import Path
remote, readme, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(readme, "LEEME.txt")
    for path in sorted(p for p in remote.rglob("*") if p.is_file()):
        archive.write(path, Path("dvc-respaldo") / path.relative_to(remote))
print(f"ZIP con {len(archive.namelist())} archivos")
EOF
}

probar() {
    local test_dir
    test_dir="$BASE/prueba-respaldo-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$test_dir"
    git clone --quiet "$REPO_URL" "$test_dir/repo"
    "$PY" -m zipfile -e "$ZIP" "$test_dir/remoto"

    cd "$test_dir/repo"
    "$DVC" remote add -d respaldo "$test_dir/remoto/dvc-respaldo"
    "$DVC" pull
    "$DVC" status
    # Se usa el código del clon, no el del repo de trabajo.
    PYTHONPATH="$test_dir/repo/src" "$PY" -m dsrv.verify_data --out "$test_dir/salida"

    "$PY" - "$ZIP" "$test_dir" "$EVIDENCE" <<'EOF'
import hashlib, json, subprocess, sys, zipfile
from pathlib import Path
zip_path, test_dir, evidence = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
names = zipfile.ZipFile(zip_path).namelist()
digest = hashlib.sha256()
with zip_path.open("rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
        digest.update(chunk)
repo = test_dir / "repo"
data = json.loads((test_dir / "salida" / "verificacion_datos.json").read_text(encoding="utf-8"))
commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True)
report = {
    "motivo": "DagsHub está privado por una caída del servicio; se hará público apenas funcione.",
    "zip": zip_path.name,
    "zip_sha256": digest.hexdigest(),
    "zip_bytes": zip_path.stat().st_size,
    "archivos_en_zip": len(names),
    "objetos_dvc_en_zip": sum(1 for name in names if name.startswith("dvc-respaldo/") and not name.endswith("/")),
    "prueba": {
        "entorno": "clon nuevo desde GitHub en una carpeta nueva, con dvc pull solo desde el ZIP",
        "commit_clonado": commit.stdout.strip(),
        "verificacion_datos": data,
        "resultado": data["resultado"],
    },
}
evidence.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Prueba del respaldo: {data['resultado']}")
EOF
}

case "${1:-}" in
    crear) crear ;;
    probar) probar ;;
    *) echo "uso: bash scripts/dvc_backup.sh crear|probar" >&2; exit 2 ;;
esac
