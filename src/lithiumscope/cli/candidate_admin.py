"""Explicit non-interactive access to the preserved candidate operations."""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("bundle", help="Prepare a local release bundle; does not publish")
    commands.add_parser("evaluate", help="Evaluate candidates without promotion")
    promote = commands.add_parser("promote", help="Blocked until independent holdout validation is available")
    promote.add_argument("--confirm", action="store_true", required=True)
    args = parser.parse_args(argv)
    if args.command == "bundle":
        from lithiumscope.results.release import prepare_release_candidate_bundle

        manifest, bundle = prepare_release_candidate_bundle()
        print(f"Manifest: {manifest}\nBundle: {bundle}")
        return 0
    from lithiumscope.prediction.candidate_evaluation import evaluate_pending_candidates

    if args.command == "promote":
        print("Promoción bloqueada: se requiere un holdout independiente.")
        return 2
    result = evaluate_pending_candidates(promote_if_better=False)
    print(f"Evaluación: {result.evaluation_id}\nManifest: {result.manifest_path}")
    for group, decision in result.decisions.items():
        print(f"{group}: {decision.get('status', 'N/D')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
