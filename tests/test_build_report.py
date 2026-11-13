import re

import pytest

from dsrv import build_report as br


def test_numbers_use_spanish_format():
    assert br.num(0.903518, 4) == "0,9035"
    assert br.num(1234.5, 1) == "1.234,5"
    assert br.pct(0.681) == "68,1 %"
    assert br.count(175341) == "175.341"
    assert br.points(-0.0732) == "−7,32"
    assert br.points(0.1066) == "+10,66"


def test_inline_html_escapes_and_formats():
    text = br._inline_html("**Nota:** `a<b` y [enlace](https://x.test) *cursiva*")

    assert "<strong>Nota:</strong>" in text
    assert "<code>a&lt;b</code>" in text
    assert '<a href="https://x.test">enlace</a>' in text
    assert "<em>cursiva</em>" in text


BLOCKS = [
    ("title", "Título"),
    ("h2", "1. Sección"),
    ("p", "Texto con **negrita**."),
    ("table", ["A", "B"], [["x", "1"], ["y", "2"]], "lr", "small"),
    ("ul", ["uno", "dos"]),
    ("fig", "figura.png", "Pie de figura."),
    ("break",),
    ("h2", "Referencias"),
]


def test_markdown_and_html_render_the_same_blocks():
    markdown = br.render_markdown(BLOCKS, "figuras/figura.png")
    page = br.render_html(BLOCKS, b"\x89PNG")

    assert "| A | B |" in markdown and "|---|---:|" in markdown
    assert "![Pie de figura.](figuras/figura.png)" in markdown
    assert '<table class="small">' in page and '<td class="r">2</td>' in page
    assert "data:image/png;base64," in page
    assert page.index('class="break"') < page.index("Referencias")


@pytest.fixture(scope="module")
def generated():
    evidence = br.load_evidence()
    manual = {
        "integrantes": "MARCADOR_INTEGRANTES",
        "declaracion_uso_ia": "MARCADOR_IA",
        "contribucion_integrantes": "MARCADOR_CONTRIBUCION",
    }
    blocks = br.build_blocks(evidence, manual, br.FIGURE_PATH)
    return evidence, br.render_markdown(blocks, "figuras/umbral_v2.png")


def test_report_has_the_required_sections_and_markers(generated):
    _, markdown = generated
    headings = re.findall(r"^## (.+)$", markdown, flags=re.MULTILINE)

    assert headings == [
        "1. Datos y limpieza",
        "2. Cuatro experimentos",
        "3. Cambios entre v1 y v2",
        "4. Elección del umbral",
        "5. Test final",
        "6. Limitaciones",
        "Declaración de uso de IA",
        "Contribución de cada integrante",
        "Referencias",
    ]
    for marker in ("MARCADOR_INTEGRANTES", "MARCADOR_IA", "MARCADOR_CONTRIBUCION"):
        assert marker in markdown


def test_report_figures_come_from_the_evidence(generated):
    evidence, markdown = generated
    splits = evidence["audit"]["splits"]

    assert br.count(splits["filas_eliminadas"]["duplicado_exacto"]) in markdown
    assert br.count(splits["filas_eliminadas"]["grupo_contradictorio"]) in markdown
    assert br.pct(evidence["test"]["fpr"], 2) in markdown
    assert evidence["backup"]["zip_sha256"][:16] in markdown
    assert not re.search(r"\bnan\b|\bNone\b|\{|\}", markdown)
