import argparse
import csv
import json

from autosynth.common import write_json
from autosynth.evaluation import report
from autosynth.paths import resolve_project_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--distribution-summary")
    args = parser.parse_args()
    merged = None
    fields = set()
    metrics = {
        "clap",
        "languagebind",
        "wmfcc",
        "sot",
        "mss",
        "rms",
        "ce",
        "cu",
        "pq",
        "kl",
    }
    for path in args.metrics:
        with resolve_project_path(path).open(
            newline="", encoding="utf-8-sig"
        ) as stream:
            rows = list(csv.DictReader(stream))
        rows = {(r.get("dataset", "test"), r["id"]): r for r in rows}
        if merged is None:
            merged = rows
        elif rows.keys() != merged.keys():
            raise ValueError("Metric files must contain the same samples")
        for key, row in rows.items():
            for name in ("output", "reference", "prompt"):
                if (
                    name in row
                    and name in merged[key]
                    and row[name] != merged[key][name]
                ):
                    raise ValueError(f"Metric file mismatch for {key}: {name}")
            merged[key].update(
                {name: float(value) for name, value in row.items() if name in metrics}
            )
        fields.update(metrics.intersection(next(iter(rows.values()))))
    report(list(merged.values()), args.output, sorted(fields))
    if args.distribution_summary:
        path = resolve_project_path(args.output) / "summary.json"
        summary = json.loads(path.read_text())
        distribution = json.loads(
            resolve_project_path(args.distribution_summary).read_text()
        )
        for dataset, values in distribution.items():
            if dataset not in summary or summary[dataset]["count"] != values["count"]:
                raise ValueError("Distribution metric sample count mismatch")
            summary[dataset].update({key: values[key] for key in ("fad", "kl")})
        write_json(path, summary)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
