import csv
import gzip
import os

csv_files = [
    "articles.csv",
    "customers.csv",
    "transactions_train.csv",
    "sample_submission.csv"
]


def open_csv(file):
    # Usa la versión comprimida si no está el CSV original.
    if not os.path.exists(file) and os.path.exists(file + ".gz"):
        return gzip.open(file + ".gz", "rt", encoding="utf-8", newline="")
    return open(file, "r", encoding="utf-8", newline="")


for file in csv_files:
    print("\n" + "=" * 80)
    print(f"ARCHIVO: {file}")
    print("=" * 80)

    # Contar filas
    with open_csv(file) as f:
        total_lines = sum(1 for _ in f)

    num_rows = total_lines - 1  # quitar cabecera

    # Leer cabecera y primera fila
    with open_csv(file) as f:
        reader = csv.reader(f)

        header = next(reader)
        first_row = next(reader, None)

    print(f"Número de filas: {num_rows:,}".replace(",", "."))

    print("\nCABECERAS:")
    for i, column in enumerate(header):
        print(f"  {i}: {column}")

    print("\nPRIMERA FILA:")
    if first_row:
        for column, value in zip(header, first_row):
            print(f"  {column}: {value}")
    else:
        print("  El CSV no contiene datos.")
