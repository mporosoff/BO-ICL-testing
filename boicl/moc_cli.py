"""Local command-line access to the same MoC campaign services as the browser."""
import argparse
import gzip
import json
from pathlib import Path
import socket

from .campaign import CampaignService, atomic_json


def offline_demo(output, sampler_settings=None):
    from .moc_demo import demo_runner

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    service = CampaignService(output / "state", runner=demo_runner)
    overrides = {"auto_suggest": False, "new_measurement_budget": 3}
    if sampler_settings is not None:
        overrides["structured_gp"] = sampler_settings
    pair = service.create_pair(synthetic_demo=True, overrides=overrides)
    report = {
        "synthetic_demo": True,
        "provider_calls": 0,
        "laboratory_validation": False,
        "arms": {},
    }
    for name in ["gp", "llm"]:
        cid = pair[name]
        service.start_suggestion(cid, background=False)
        first = service.summary(cid)["suggestions"][-1]
        if first["status"] != "suggested":
            raise RuntimeError(service.summary(cid)["progress"])
        service.reserve(cid, first["suggestion_id"])
        service.measure(
            cid,
            first["suggestion_id"],
            dict(
                moc_wt_pct=75.0,
                moc_wt_pct_sigma=1.5,
                gof=1.0,
                closure_gap_wt_pct=0.0,
                closure_gap_origin="explicit synthetic fixture",
                source_note="SYNTHETIC; not a laboratory result",
            ),
            request_id="synthetic-first-measurement",
            refresh=False,
        )
        service.start_suggestion(cid, background=False)
        second = service.summary(cid)["suggestions"][-1]
        if second["status"] != "suggested":
            raise RuntimeError(service.summary(cid)["progress"])
        service.reserve(cid, second["suggestion_id"])
        bundle = service.export(cid)
        raw = json.dumps(bundle, ensure_ascii=False, allow_nan=False).encode("utf-8")
        (output / f"{name}-campaign.json.gz").write_bytes(gzip.compress(raw, mtime=0))
        replay = service.replay(cid, first["suggestion_id"])
        resumed = CampaignService(output / f"{name}-resume")
        restored = resumed.import_bundle(
            json.loads(
                gzip.decompress((output / f"{name}-campaign.json.gz").read_bytes())
            )
        )
        counts = resumed.summary(restored)["counts"]
        if counts["measured"] != 4 or counts["pending"] != 1:
            raise AssertionError("Resume state mismatch")
        report["arms"][name] = {
            "initial_candidate": first["candidate_id"],
            "next_candidate": second["candidate_id"],
            "counts_after_resume": counts,
            "replay_verified": replay.get("replay_verified", True),
            "selection_reason": first["selection_reason"],
        }
    atomic_json(output / "matched-manifest.json", pair["manifest"])
    atomic_json(output / "verification.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", default=".moc-campaigns")
    subs = parser.add_subparsers(dest="command", required=True)
    demo = subs.add_parser("demo")
    demo.add_argument("--output", default=".moc-demo/offline-export")
    init = subs.add_parser("init")
    init.add_argument("--preset", default="moc_gp")
    init.add_argument("--pair", action="store_true")
    suggest = subs.add_parser("suggest")
    suggest.add_argument("campaign_id")
    snapshot = subs.add_parser("snapshot")
    snapshot.add_argument("campaign_id")
    export = subs.add_parser("export")
    export.add_argument("campaign_id")
    export.add_argument("output")
    replay = subs.add_parser("replay")
    replay.add_argument("campaign_id")
    replay.add_argument("suggestion_id")
    args = parser.parse_args(argv)
    if args.command == "demo":
        original = socket.socket.connect

        def no_network(*a, **kw):
            raise RuntimeError("Offline demonstration prohibits network requests")

        socket.socket.connect = no_network
        try:
            result = offline_demo(args.output)
        finally:
            socket.socket.connect = original
    else:
        svc = CampaignService(args.state_dir)
        if args.command == "init":
            result = (
                svc.create_pair()
                if args.pair
                else {"campaign_id": svc.create(args.preset)}
            )
        elif args.command == "suggest":
            svc.start_suggestion(args.campaign_id, background=False)
            result = svc.summary(args.campaign_id)
        elif args.command == "snapshot":
            result = svc.summary(args.campaign_id)
        elif args.command == "export":
            atomic_json(args.output, svc.export(args.campaign_id))
            result = {"exported": args.output}
        else:
            result = svc.replay(args.campaign_id, args.suggestion_id)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
