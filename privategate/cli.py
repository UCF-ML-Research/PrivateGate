from __future__ import annotations

import argparse
import json
import sys
from typing import Optional


def _cmd_rewrite(args: argparse.Namespace) -> int:
    from privategate.detector.rule_detector import RuleDetector
    from privategate.policy.engine import PolicyEngine
    from privategate.policy.table import load_default_policy
    from privategate.rewriter.rewriter import Rewriter

    detector = RuleDetector()
    engine = PolicyEngine(load_default_policy())
    rewriter = Rewriter()

    spans = detector.detect(args.text)
    decisions = engine.decide(spans)
    result = rewriter.rewrite(args.text, decisions)

    print(json.dumps({
        "transformed": result.transformed_text,
        "placeholder_map": result.placeholder_map,
        "has_secure_slots": result.has_secure_slots,
        "decisions": [
            {
                "span": d.span.text,
                "category": d.span.category.value,
                "risk": d.span.risk.value,
                "action": d.action.value,
            }
            for d in result.decisions
        ],
    }, indent=2))
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from privategate.detector.rule_detector import RuleDetector
    from privategate.policy.engine import PolicyEngine
    from privategate.policy.table import load_default_policy
    from privategate.rewriter.rewriter import Rewriter
    from privategate.backends.mock_standard import MockStandardBackend
    from privategate.backends.mock_secure import MockSecureBackend
    from privategate.router.router import Router
    from privategate.reconstruct.reconstructor import Reconstructor

    detector = RuleDetector()
    engine = PolicyEngine(load_default_policy())
    rewriter = Rewriter()
    router = Router()
    reconstructor = Reconstructor()

    spans = detector.detect(args.text)
    decisions = engine.decide(spans)
    rw = rewriter.rewrite(args.text, decisions)
    routing = router.route(rw, probe_results=[])

    backend = MockSecureBackend() if routing.path == "secure" else MockStandardBackend()
    raw = backend.complete(rw.transformed_text)
    final = reconstructor.reconstruct(raw, rw.placeholder_map)

    print(json.dumps({
        "answer": final,
        "routing": routing.path,
        "reason": routing.reason,
    }, indent=2))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="privategate")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rw = sub.add_parser("rewrite", help="rewrite a query into its protected form")
    p_rw.add_argument("text")
    p_rw.set_defaults(func=_cmd_rewrite)

    p_ask = sub.add_parser("ask", help="run the full pipeline against a backend")
    p_ask.add_argument("text")
    p_ask.set_defaults(func=_cmd_ask)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
