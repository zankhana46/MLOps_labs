# Apache Beam Labs
Apache Beam lab: parse CSV, aggregate and write results.

This lab demonstrates a simple Apache Beam pipeline you can run locally from VS Code.
It reads a CSV of “transactions,” cleans the data, computes:

Total revenue per category, and

Top-N products by revenue within each category,

then writes neat CSV outputs.

### What the pipeline does

1. Read: data/transactions.csv (single file, first line is the header).

2. Parse & Clean:

Skip blank/header lines.

Validate numeric fields: price (float) and quantity (int).

Compute revenue = price * quantity.

Drop bad rows (e.g., missing quantity).

3. Aggregate:

category_totals.csv: total revenue per category.

topN_products_by_category.csv: top-N products (by revenue) within each category.

4. Write: outputs are single CSV files (forced to num_shards=1).
