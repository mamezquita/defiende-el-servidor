#!/usr/bin/env bash
# Prueba de recuperación en un entorno nuevo: clon, venv, datos de DVC y modelos de Hugging Face.
#
# Uso:
#   read -rsp "Token de DagsHub: " DAGSHUB_TOKEN; echo; export DAGSHUB_TOKEN
#   bash scripts/recovery_test.sh --commit <SHA> --sha-v1 <SHA> --sha-v2 <SHA>
#
# Opcionales: --repo-url, --hf-repo y --workdir (por defecto ~/mlops/recuperacion-<fecha-hora>).
# El token solo se usa en el remote de DVC del clon: no se imprime y se borra al terminar.

set -euo pipefail

REPO_URL="https://github.com/mamezquita/defiende-el-servidor.git"
HF_REPO="MiguelAngelAB/unsw-nb15-defiende-servidor"
COMMIT=""
SHA_V1=""
SHA_V2=""
WORKDIR=""

die() {
    echo "ERROR: $*" >&2
    exit 2
}

while [ $# -gt 0 ]; do
    [ $# -ge 2 ] || die "falta el valor de $1"
    case "$1" in
        --repo-url) REPO_URL=$2 ;;
        --commit) COMMIT=$2 ;;
        --hf-repo) HF_REPO=$2 ;;
        --sha-v1) SHA_V1=$2 ;;
        --sha-v2) SHA_V2=$2 ;;
        --workdir) WORKDIR=$2 ;;
        *) die "parámetro desconocido: $1" ;;
    esac
    shift 2
done

[ -n "$COMMIT" ] || die "falta --commit"
[ -n "$SHA_V1" ] || die "falta --sha-v1"
[ -n "$SHA_V2" ] || die "falta --sha-v2"
[ -n "${DAGSHUB_TOKEN:-}" ] || die "define la variable de entorno DAGSHUB_TOKEN"

WORKDIR=${WORKDIR:-"$HOME/mlops/recuperacion-$(date +%Y%m%d-%H%M%S)"}
[ ! -e "$WORKDIR" ] || die "$WORKDIR ya existe; usa una carpeta nueva"
mkdir -p "$WORKDIR/salida"
WORKDIR=$(cd "$WORKDIR" && pwd)

OUTPUT="$WORKDIR/salida"
REPO_DIR="$WORKDIR/repo"
LOG="$WORKDIR/recovery_log.txt"
COMMANDS="$OUTPUT/comandos.txt"
TIMES="$OUTPUT/tiempos.csv"
STARTED=$(date -Iseconds)
CURRENT_STEP=""
STEP_START=0

# Un entorno heredado podría ocultar que la recuperación depende de algo externo.
unset PYTHONPATH VIRTUAL_ENV HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_TOKEN_PATH
echo "paso,segundos,estado" > "$TIMES"
: > "$COMMANDS"

exec 3>&1 4>&2
exec > >(tee -a "$LOG") 2>&1
TEE_PID=$!

log_run() {
    printf '%q ' "$@" >> "$COMMANDS"
    printf '\n' >> "$COMMANDS"
    "$@"
}

escape_sed() {
    printf '%s' "$1" | sed 's/[][\.*^$/&]/\\&/g'
}

finish() {
    local status=$?
    set +e

    if [ -n "$CURRENT_STEP" ]; then
        printf '%s,%s,fallo\n' "$CURRENT_STEP" "$((SECONDS - STEP_START))" >> "$TIMES"
    fi
    rm -f "$REPO_DIR/.dvc/config.local"

    {
        printf 'git_commit %s\n' "$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null)"
        printf 'sistema %s\n' "$(uname -sr)"
        if [ -x "$REPO_DIR/.venv/bin/python" ]; then
            "$REPO_DIR/.venv/bin/python" --version
            "$REPO_DIR/.venv/bin/python" -m pip freeze --exclude-editable
        fi
    } > "$OUTPUT/entorno.txt" 2>&1

    local state="OK"
    [ "$status" -eq 0 ] || state="FALLA"
    printf '{\n  "estado": "%s",\n  "repo_url": "%s",\n  "commit": "%s",\n  "hf_repo": "%s",\n  "sha_v1": "%s",\n  "sha_v2": "%s",\n  "inicio": "%s",\n  "fin": "%s"\n}\n' \
        "$state" "$REPO_URL" "$COMMIT" "$HF_REPO" "$SHA_V1" "$SHA_V2" "$STARTED" "$(date -Iseconds)" \
        > "$OUTPUT/resumen_ejecucion.json"
    echo "=== Fin: $state"

    exec 1>&3 2>&4
    wait "$TEE_PID" 2>/dev/null
    cp "$LOG" "$OUTPUT/recovery_log.txt"

    # Rutas y usuario de esta máquina no deben quedar en las evidencias.
    local file
    for file in "$OUTPUT"/*; do
        case "$file" in
            *.txt | *.json | *.csv)
                sed -i \
                    -e "s/$(escape_sed "$WORKDIR")/<WorkDir>/g" \
                    -e "s/$(escape_sed "$HOME")/<HOME>/g" \
                    -e "s/$(escape_sed "$(id -un)")/<USER>/g" \
                    "$file"
                ;;
        esac
    done

    if grep -rqlF -- "$DAGSHUB_TOKEN" "$OUTPUT" "$LOG"; then
        echo "ERROR: el valor del token aparece en estos archivos; NO los compartas:" >&2
        grep -rlF -- "$DAGSHUB_TOKEN" "$OUTPUT" "$LOG" | sed "s#$(escape_sed "$WORKDIR")#<WorkDir>#" >&2
        status=3
    fi

    echo "Estado: $state. Salida en: $OUTPUT" >&3
    exit "$status"
}
trap finish EXIT

run_step() {
    CURRENT_STEP=$1
    STEP_START=$SECONDS
    shift
    echo "=== Paso $CURRENT_STEP"
    "$@"
    printf '%s,%s,ok\n' "$CURRENT_STEP" "$((SECONDS - STEP_START))" >> "$TIMES"
    CURRENT_STEP=""
}

step_clone() {
    log_run git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
    log_run git checkout --detach "$COMMIT"
    log_run git rev-parse HEAD
    log_run git --no-pager show v1 --stat
    log_run git --no-pager show v2 --stat
}

step_environment() {
    log_run python3.11 -m venv .venv
    log_run .venv/bin/python --version
    log_run .venv/bin/python -m pip install --progress-bar off -r requirements.txt
    log_run .venv/bin/python -m pip install --progress-bar off -e .
}

step_data() {
    # Se registra sin expandir: el valor del token no debe llegar a comandos.txt.
    printf '%s\n' '.venv/bin/dvc remote modify origin --local access_key_id "$DAGSHUB_TOKEN"' >> "$COMMANDS"
    .venv/bin/dvc remote modify origin --local access_key_id "$DAGSHUB_TOKEN"
    printf '%s\n' '.venv/bin/dvc remote modify origin --local secret_access_key "$DAGSHUB_TOKEN"' >> "$COMMANDS"
    .venv/bin/dvc remote modify origin --local secret_access_key "$DAGSHUB_TOKEN"

    log_run .venv/bin/dvc pull
    log_run .venv/bin/dvc status
    log_run .venv/bin/python -m dsrv.verify_data --out "$OUTPUT"
}

step_hugging_face_cache() {
    export HF_HOME="$WORKDIR/hf_home"
    printf 'export HF_HOME=%q\n' "$HF_HOME" >> "$COMMANDS"
    log_run mkdir -p "$HF_HOME"
}

step_recover() {
    local inputs=evidencias/recuperacion/entradas_fijas.csv
    (cd evidencias/recuperacion && log_run sha256sum -c entradas_fijas.csv.sha256)
    log_run .venv/bin/python -m dsrv.recover --version v1 --sha "$SHA_V1" --hf-repo "$HF_REPO" \
        --inputs "$inputs" --reference evidencias/recuperacion/predicciones_originales_v1.csv \
        --out "$OUTPUT"
    log_run .venv/bin/python -m dsrv.recover --version v2 --sha "$SHA_V2" --hf-repo "$HF_REPO" \
        --inputs "$inputs" --reference evidencias/recuperacion/predicciones_originales_v2.csv \
        --out "$OUTPUT"
}

run_step "clon_y_tags" step_clone
run_step "entorno" step_environment
run_step "datos_dvc" step_data
run_step "cache_hugging_face" step_hugging_face_cache
run_step "recuperacion_modelos" step_recover
