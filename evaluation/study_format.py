"""Plain-text tables for the write-up. The test checks these strings are quoted."""

from __future__ import annotations

from data.cities import STUDY_CITIES

from evaluation.study_protocol import ELIGIBLE_MODELS, MODEL_NAMES


def money(value: float) -> str:
    return f"${value:,.2f}"


def r2(value: float) -> str:
    return f"{value:.3f}"


def pct(value: float) -> str:
    return f"{value:.1f}%"


def nights_mae(value: float) -> str:
    return f"{value:.1f}"


def price_comparison_table(metrics: dict) -> str:
    """Holdout MAE by model. The average gives each city equal weight."""
    names = list(ELIGIBLE_MODELS)
    header = "| Model | " + " | ".join(metrics["cities"][city]["display"] for city in STUDY_CITIES)
    header += " | Average |"
    rule = "| --- | " + " | ".join("---:" for _ in STUDY_CITIES) + " | ---: |"
    lines = [header, rule]
    for name in names:
        cells = [MODEL_NAMES[name]]
        values = []
        for city in STUDY_CITIES:
            mae = metrics["cities"][city]["price"]["holdout"][name]["mae"]
            values.append(mae)
            cells.append(money(mae))
        cells.append(money(sum(values) / len(values)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def price_result_table(metrics: dict) -> str:
    """Chosen model on the primary holdout, with cell-bootstrap intervals."""
    lines = [
        "| City | Listings | Cells | R² | R² interval | MAE | MAE interval | Median % error | Median % interval | Local median MAE | Error removed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for city in STUDY_CITIES:
        block = metrics["cities"][city]
        hold = block["price"]["holdout"][metrics["shipped_model"]]
        boot = block["price"]["bootstrap"]
        lines.append(
            "| "
            + " | ".join(
                [
                    block["display"],
                    f"{hold['n']:,}",
                    f"{block['price']['n_test_cells']:,}",
                    r2(hold["r2"]),
                    f"{r2(boot['r2']['low'])} to {r2(boot['r2']['high'])}",
                    money(hold["mae"]),
                    f"{money(boot['mae']['low'])} to {money(boot['mae']['high'])}",
                    pct(hold["median_ape"]),
                    f"{pct(boot['median_ape']['low'])} to {pct(boot['median_ape']['high'])}",
                    money(block["price"]["holdout"]["area_median"]["mae"]),
                    pct(100.0 * boot["error_removed"]["point"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def price_confirmation_table(metrics: dict) -> str:
    lines = [
        "| City | Primary R² | Other seeds R² | Primary MAE | Other seeds MAE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for city in STUDY_CITIES:
        block = metrics["cities"][city]["price"]["confirmation"]
        r2s = ", ".join(r2(value) for value in block["r2"])
        maes = ", ".join(money(value) for value in block["mae"])
        lines.append(
            "| "
            + " | ".join(
                [
                    metrics["cities"][city]["display"],
                    r2(block["primary_r2"]),
                    f"{r2s} (spread {r2(block['r2_spread'])})",
                    money(block["primary_mae"]),
                    f"{maes} (spread {money(block['mae_spread'])})",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def nights_comparison_table(metrics: dict) -> str:
    header = "| Model | " + " | ".join(metrics["cities"][city]["display"] for city in STUDY_CITIES)
    header += " | Average |"
    rule = "| --- | " + " | ".join("---:" for _ in STUDY_CITIES) + " | ---: |"
    lines = [header, rule]
    for name in ELIGIBLE_MODELS:
        # The price-table name calls this baseline a rate. Here it is a nights count.
        label = "Local median nights" if name == "area_median" else MODEL_NAMES[name]
        cells = [label]
        values = []
        for city in STUDY_CITIES:
            mae = metrics["cities"][city]["nights"]["holdout"][name]["mae"]
            values.append(mae)
            cells.append(nights_mae(mae))
        cells.append(nights_mae(sum(values) / len(values)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def nights_result_table(metrics: dict) -> str:
    lines = [
        "| City | R² | R² interval | MAE (nights) | MAE interval | Median % error | Local median MAE | Error removed | Cleared |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    shipped = metrics["nights_shipped_model"]
    for city in STUDY_CITIES:
        block = metrics["cities"][city]
        hold = block["nights"]["holdout"][shipped]
        boot = block["nights"]["bootstrap"]
        lines.append(
            "| "
            + " | ".join(
                [
                    block["display"],
                    r2(hold["r2"]),
                    f"{r2(boot['r2']['low'])} to {r2(boot['r2']['high'])}",
                    nights_mae(hold["mae"]),
                    f"{nights_mae(boot['mae']['low'])} to {nights_mae(boot['mae']['high'])}",
                    pct(hold["median_ape"]),
                    nights_mae(block["nights"]["holdout"]["area_median"]["mae"]),
                    pct(100.0 * boot["error_removed"]["point"]),
                    "Yes" if block["nights"]["cleared_bar"] else "No",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def nights_confirmation_table(metrics: dict) -> str:
    lines = [
        "| City | Primary R² | Other seeds R² | Primary MAE | Other seeds MAE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for city in STUDY_CITIES:
        block = metrics["cities"][city]["nights"]["confirmation"]
        r2s = ", ".join(r2(value) for value in block["r2"])
        maes = ", ".join(nights_mae(value) for value in block["mae"])
        lines.append(
            "| "
            + " | ".join(
                [
                    metrics["cities"][city]["display"],
                    r2(block["primary_r2"]),
                    f"{r2s} (spread {r2(block['r2_spread'])})",
                    nights_mae(block["primary_mae"]),
                    f"{maes} (spread {nights_mae(block['mae_spread'])})",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def cleaning_table(reports: dict) -> str:
    from data.clean import render_quality_markdown

    return render_quality_markdown(reports)
