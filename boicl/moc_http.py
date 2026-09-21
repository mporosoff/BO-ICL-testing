"""MoC endpoints for the existing local HTTP server (no separate framework)."""
import base64
from copy import deepcopy
import json
from pathlib import Path
import threading
from urllib.parse import parse_qs

from .campaign import (
    CampaignService,
    uid,
    now,
    embedding_spec,
    embedding_cache_directory,
)
from .moc_import import load_moc_package


def service(handler):
    cls = type(handler)
    with cls.moc_init_lock:
        if getattr(cls, "moc_service", None) is None:
            demo = getattr(cls, "moc_demo", False)
            if demo:
                from .moc_demo import demo_runner

                runner = demo_runner
            else:
                runner = None
            cls.moc_service = CampaignService(
                handler.state.root / (".moc-demo" if demo else ".moc-campaigns"),
                runner=runner,
            )
        return cls.moc_service


def get(handler, parsed):
    svc = service(handler)
    query = parse_qs(parsed.query)
    cid = query.get("id", [""])[0]
    action = parsed.path.rsplit("/", 1)[-1]
    if action == "list":
        handler._send_json(
            {
                "campaigns": svc.list(),
                "demo": bool(getattr(type(handler), "moc_demo", False)),
            }
        )
    elif action == "state":
        handler._send_json(svc.summary(cid))
    elif action == "checkpoints":
        handler._send_json(
            {"campaign_id": cid, "checkpoints": svc.list_checkpoints(cid)}
        )
    elif action == "export":
        raw = json.dumps(svc.export(cid), ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header(
            "Content-Disposition", f"attachment; filename=moc-campaign-{cid}.json"
        )
        handler.send_header("Content-Length", str(len(raw)))
        handler.end_headers()
        handler.wfile.write(raw)
    else:
        raise ValueError("Unknown MoC endpoint")


def cache_for(svc, cid):
    from .embedding_cache import create_embedding_cache

    data = svc.get(cid)
    engine = data["config"]["engine"]
    if engine == "gpr_features":
        raise ValueError("Synthesis-parameter GP needs no embeddings")
    spec = embedding_spec(data["config"])
    return create_embedding_cache(
        embedding_cache_directory(svc.root, data["config"]), spec
    )


def prepare_cache(svc, cid):
    from .request_policy import RequestPolicy, ReliableClient

    with svc.lock:
        data = deepcopy(svc.get(cid))
        current = svc.jobs.get(cid)
        if data["synthetic_demo"]:
            raise ValueError(
                "Demo vectors are synthetic; production cache generation is unavailable in demo mode"
            )
        if current and current["status"] == "running":
            return {"coalesced": True, "job_id": current["job_id"]}
        cache = cache_for(svc, cid)
        job = {
            "job_id": uid(),
            "status": "running",
            "detail": "Validating cache",
            "cancel": threading.Event(),
            "started_at": now(),
        }
        svc.jobs[cid] = job

    def run():
        try:
            from openai import OpenAI

            log = []
            client = None
            policy = RequestPolicy(data["config"]["api"], job["cancel"], log)
            job["provider_attempts"] = log

            def embedder(texts):
                nonlocal client
                if client is None:
                    client = ReliableClient(OpenAI(max_retries=0), policy)
                return client.embeddings.create(model=cache.spec.model, input=texts)

            def progress(report):
                job[
                    "detail"
                ] = f"Generated {report['generated']} embeddings; {len(report['errors'])} errors"

            report = cache.prepare(
                data["candidates"],
                embedder,
                cancelled=job["cancel"].is_set,
                progress=progress,
            )
            job.update(
                status="cancelled"
                if report.get("cancelled")
                else "complete"
                if not report["missing_ids"]
                else "failed",
                detail=f"Validated {report['hit_count']} / {report['requested']} embeddings; generated {report['generated']}; {len(report['errors'])} errors",
                cache_report=report,
                completed_at=now(),
            )
        except Exception as exc:
            job.update(status="failed", detail=str(exc), completed_at=now())

    thread = threading.Thread(target=run, daemon=True)
    job["thread"] = thread
    thread.start()
    return {"job_id": job["job_id"]}


def post(handler, parsed):
    svc = service(handler)
    p = handler._read_json()
    if not isinstance(p, dict):
        raise ValueError("MoC requests must be JSON objects")
    cid = p.get("id")
    action = parsed.path.rsplit("/", 1)[-1]
    if action in {"create", "pair"}:
        package = (
            load_moc_package(
                workbook_bytes=base64.b64decode(p["workbook"], validate=True)
            )
            if p.get("workbook")
            else load_moc_package()
        )
        options = {
            "package": package,
            "overrides": {"new_measurement_budget": p.get("budget")},
            "synthetic_demo": bool(getattr(type(handler), "moc_demo", False)),
        }
        result = (
            svc.create_pair(**options)
            if action == "pair"
            else {"campaign_id": svc.create(p.get("preset", "moc_llm"), **options)}
        )
    elif action == "suggest":
        result = svc.start_suggestion(cid)
    elif action == "checkpoint":
        result = svc.save_checkpoint(cid, p.get("name"))
    elif action == "restore-checkpoint":
        result = {
            "campaign_id": svc.restore_checkpoint(
                cid, p["checkpoint_id"], p.get("name")
            )
        }
    elif action == "cancel":
        result = svc.cancel_job(cid)
    elif action == "reserve":
        result = svc.reserve(cid, p["suggestion_id"])
    elif action == "release":
        result = svc.cancel_reservation(cid, p["suggestion_id"])
    elif action == "measure":
        result = svc.measure(cid, p["suggestion_id"], p["values"], p.get("request_id"))
    elif action == "refine":
        result = svc.refine(cid, p["observation_id"], p["values"], p["reason"])
    elif action in {"config", "config-preview"}:
        result = svc.update_config(cid, p["changes"], apply=action == "config")
    elif action == "import":
        if "bundle_json" in p:
            if "bundle" in p or not isinstance(p["bundle_json"], str):
                raise ValueError("Supply one campaign bundle as JSON text or an object")
            # Browser JSON numbers cannot represent NumPy's 128-bit RNG state.
            # Decode the original file here; JSON text preserves every integer.
            bundle = json.loads(p["bundle_json"])
        else:
            bundle = p["bundle"]
        if not isinstance(bundle, dict):
            raise ValueError("Campaign bundle must be a JSON object")
        demo = bool(getattr(type(handler), "moc_demo", False))
        if bundle.get("synthetic_demo") is not demo:
            raise ValueError(
                "Synthetic demonstration and live campaign bundles must be opened in their matching runner mode"
            )
        result = {"campaign_id": svc.import_bundle(bundle)}
    elif action == "replay":
        result = svc.replay(cid, p["suggestion_id"])
    elif action == "cache-import":
        with svc.lock:
            if svc.jobs.get(cid, {}).get("status") == "running":
                raise ValueError(
                    "Wait for or cancel current work before importing a cache"
                )
            result = cache_for(svc, cid).import_package(
                Path(p["path"]), svc.get(cid)["candidates"]
            )
    elif action == "cache-prepare":
        result = prepare_cache(svc, cid)
    else:
        raise ValueError("Unknown MoC action")
    handler._send_json(result)
