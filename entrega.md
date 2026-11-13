# Entrega: Defiende el servidor

## Qué se entrega

| Qué | Dónde |
|---|---|
| Informe | `informe/informe.pdf` |
| Código y evidencias | [github.com/mamezquita/defiende-el-servidor](https://github.com/mamezquita/defiende-el-servidor) |
| Modelos v1 y v2 | [huggingface.co/MiguelAngelAB/unsw-nb15-defiende-servidor](https://huggingface.co/MiguelAngelAB/unsw-nb15-defiende-servidor) |
| Datos (remote DVC) | [dagshub.com/mamezquita/defiende-el-servidor](https://dagshub.com/mamezquita/defiende-el-servidor), privado por ahora |
| Respaldo de los datos | `dvc-remote-respaldo.zip` |

| Versión | Tag de código | Revisión de Hugging Face | SHA-256 del modelo |
|---|---|---|---|
| v1 | `v1` | `6010585cc8729c73d88d5b0d9acb360e0065a351` | `2a968766d5e661ebfe05b174370c5272fc99bdd0befeafc4d17cdc2e365d98ab` |
| v2 | `v2` | `d0375af37f9abf49930b4ec76b7430446098f1f9` | `186e9c50ef5634a2135f299a555de16f0e1f76de7c5c8bbb8f43e2c1c908014f` |

## Datos: DagsHub privado y respaldo en ZIP

DagsHub está privado por una caída del servicio y se hará público apenas funcione. Mientras tanto,
los datos se recuperan con el ZIP de respaldo del remote DVC:

- **Archivo:** `dvc-remote-respaldo.zip` (32.054.039 bytes, 7 objetos DVC)
- **SHA-256:** `1f1726747a2c5dc6e1f08c7b6248c4354c8cf29f4f085290ef26bcf0481827f2`
- **Prueba:** clon nuevo desde GitHub en una carpeta nueva, con dvc pull solo desde el ZIP; resultado **OK**
  (commit `eed91a8d6e5227d38974b05bc547652cf1a2e606`). Detalle en `evidencias/respaldo-dvc.json`.

Uso, también explicado en el `LEEME.txt` del ZIP:

```bash
python3 -m zipfile -e dvc-remote-respaldo.zip <carpeta>
git clone https://github.com/mamezquita/defiende-el-servidor.git
cd defiende-el-servidor
# Instalar según el README, y luego:
dvc remote add -d respaldo <carpeta>/dvc-respaldo
dvc pull
python -m dsrv.verify_data --out verificacion
```

Archivo generado por `python -m dsrv.build_report`.
