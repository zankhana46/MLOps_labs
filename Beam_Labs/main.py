import argparse
import csv
import io
from typing import Dict, Iterable, List, Tuple

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions


# ---------- Helpers ----------
class ParseCSV(beam.DoFn):
    """
    Parses CSV lines to dicts using a known header.
    Skips the header line and blank lines.
    """
    def __init__(self, header_line: str):
        self.header_line = header_line
        self._fieldnames: List[str] = []

    def setup(self):
        # Correct way: parse the header line into a list of fieldnames.
        self._fieldnames = next(csv.reader([self.header_line]))

    def process(self, line: str) -> Iterable[Dict]:
        line = line.strip()
        if not line or line == self.header_line:
            return
        # Parse one CSV row and zip with header -> dict
        values = next(csv.reader([line]))
        # If a row has fewer columns (e.g., trailing comma), pad with empty strings
        if len(values) < len(self._fieldnames):
            values = values + [""] * (len(self._fieldnames) - len(values))
        row = dict(zip(self._fieldnames, values))
        yield row


def to_float_safe(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def to_int_safe(x: str):
    try:
        return int(x)
    except Exception:
        return None


def format_csv_line(fields: Iterable) -> str:
    out = io.StringIO()
    csv.writer(out).writerow(list(fields))
    return out.getvalue().rstrip("\r\n")


# ---------- Pipeline ----------
def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/transactions.csv",
                        help="Path to input CSV (single file).")
    parser.add_argument("--out_dir", default="outputs",
                        help="Directory for output shards.")
    parser.add_argument("--topn", type=int, default=2,
                        help="Top N products by revenue per category.")
    args, pipeline_args = parser.parse_known_args(argv)

    # Read header line once from the local file
    with open(args.input, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()

    opts = PipelineOptions(pipeline_args)

    with beam.Pipeline(options=opts) as p:
        lines = p | "ReadFile" >> beam.io.ReadFromText(args.input)

        # Parse -> Filter/Enrich with revenue
        rows = (
            lines
            | "ParseCSV" >> beam.ParDo(ParseCSV(header_line))
            | "Filter&Enrich" >> beam.ParDo(FilterAndEnrich())
        )

        # ---- 1) Total revenue by category (CSV with one header) ----
        cat_totals = (
            rows
            | "KeyByCategory" >> beam.Map(lambda r: (r["category"], r["revenue"]))
            | "SumPerCategory" >> beam.CombinePerKey(sum)
        )

        cat_totals_header = p | "CatHeader" >> beam.Create(
            [format_csv_line(("category", "total_revenue"))]
        )

        cat_totals_rows = (
            cat_totals
            | "FormatCatRows" >> beam.Map(
                lambda kv: format_csv_line((kv[0], f"{kv[1]:.2f}"))
            )
        )

        _ = ((cat_totals_header, cat_totals_rows)
             | "CatFlatten" >> beam.Flatten()
             | "WriteCategoryCSV" >> beam.io.WriteToText(
                    file_path_prefix=f"{args.out_dir}/category_totals",
                    file_name_suffix=".csv",
                    num_shards=1  # single file with header
                )
        )

        # ---- 2) Top-N products by revenue per category ----
        # Produce (category, [(product, revenue), ...]) then keep top N.
        cat_prod_revenue = (
            rows
            | "KeyByCatProd" >>
                beam.Map(lambda r: (r["category"], (r["product"], r["revenue"])))
            | "GroupByCategory" >> beam.GroupByKey()
            | "SortDesc" >> beam.Map(
                lambda kv: (kv[0], sorted(kv[1], key=lambda x: x[1], reverse=True))
            )
            | "TakeTopN" >> beam.Map(lambda kv: (kv[0], kv[1][:args.topn]))
            | "ExplodeTopN" >> beam.FlatMap(
                lambda kv: [(kv[0], prod, rev) for (prod, rev) in kv[1]]
            )
        )

        topn_header = p | "TopNHeader" >> beam.Create(
            [format_csv_line(("category", "product", "revenue"))]
        )
        topn_rows = (
            cat_prod_revenue
            | "FormatTopNRows" >> beam.Map(
                lambda t: format_csv_line((t[0], t[1], f"{t[2]:.2f}"))
            )
        )

        _ = ((topn_header, topn_rows)
             | "TopNFlatten" >> beam.Flatten()
             | "WriteTopNCSV" >> beam.io.WriteToText(
                    file_path_prefix=f"{args.out_dir}/top{args.topn}_products_by_category",
                    file_name_suffix=".csv",
                    num_shards=1
                )
        )


class FilterAndEnrich(beam.DoFn):
    """
    Keep only rows with good price & quantity,
    and compute 'revenue = price * quantity'.
    """
    def process(self, row: Dict) -> Iterable[Dict]:
        price = to_float_safe(row.get("price", ""))
        qty = to_int_safe(row.get("quantity", ""))
        if price != price or qty is None:  # NaN or missing quantity
            return
        row["price"] = price
        row["quantity"] = qty
        row["revenue"] = round(price * qty, 2)
        yield row


if __name__ == "__main__":
    run()