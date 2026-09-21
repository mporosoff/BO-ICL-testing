"""Run the actual SVG renderer offline; initialization must not stack at x=0."""
from copy import deepcopy
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

import pytest

from boicl.campaign_plot import plot_payload
from boicl.local_app import INDEX_HTML


pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node unavailable")


@pytest.fixture
def render_svg(tmp_path):
    source = re.search(
        r"function renderPlot\(\)\s*\{.*?(?=\n\s*function renderBenchmarkRuns)",
        INDEX_HTML,
        re.S,
    )
    assert source is not None, "Test must execute the production renderPlot function"
    renderer = tmp_path / "actual-render-plot.js"
    renderer.write_text(source.group(), encoding="utf-8")
    script = r"""
const fs=require('fs'),vm=require('vm');
const state=JSON.parse(fs.readFileSync(0,'utf8'));
const host={clientWidth:800,innerHTML:''};
const escapeHtml=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const context={state,$:id=>{if(id!=='plot')throw Error('Unexpected DOM access: '+id);return host;},
 sharedCampaignId:state.plot_x_axis?'shared-fixture':null,sharedCurveVisible:()=>true,
 benchmarkRunsForDisplay:()=>state.benchmark_runs||[],escapeHtml,
 fmt:value=>value==null?'—':String(Number(value))};
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
context.renderPlot();process.stdout.write(host.innerHTML);
"""

    def render(payload):
        result = subprocess.run(
            [shutil.which("node"), "-e", script, str(renderer)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=20,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        svg = ET.fromstring(result.stdout)
        assert svg.tag == "svg", result.stdout
        return svg

    return render


def shared_campaign(*, new_measurement=True, pending=True):
    rows = []
    for index, value in enumerate([72.1, 83.8, 23.4, 91]):
        if index == 3 and not new_measurement:
            continue
        rows.append(
            dict(
                candidate_id=str(index),
                observation_id=f"observation-{index}",
                physical_measurement_id=f"physical-{index}",
                moc_wt_pct=value,
                moc_wt_pct_sigma=1.5,
                is_seed=index < 3,
                record_status="measured",
                training_included=True,
            )
        )
    return dict(
        campaign_id="plot-fixture",
        pool_fingerprint="same-pool",
        initialization_fingerprint="same-three-seeds",
        config=dict(
            engine="llm",
            name="Plot fixture",
            objective="moc_wt_pct",
            units="wt%",
            direction="maximize",
            bounds=[0, 100],
            seed=616,
            repeat_policy="source_reset_quality_repeats",
            new_measurement_budget=10,
        ),
        candidates=[
            dict(candidate_id=str(i), procedure=f"recipe {i}") for i in range(6)
        ],
        observations=rows,
        suggestions=[
            dict(
                suggestion_id="pending-proposal",
                candidate_id="4",
                status="pending",
                prediction=dict(
                    mean=97,
                    sd=2,
                    lower95=92,
                    upper95=100,
                    uncertainty_type="latent-function bounded mixture",
                ),
            )
        ]
        if pending
        else [],
    )


def title(element):
    return element.findtext("title") or ""


def measured_circles(svg):
    return [node for node in svg.iter("circle") if "recipe " in title(node)]


def ticks(svg):
    return {
        node.text: float(node.attrib["x"])
        for node in svg.iter("text")
        if node.attrib.get("text-anchor") == "middle"
    }


def by_marker(svg, name):
    matches = [node for node in svg.iter() if node.attrib.get("data-" + name) == "true"]
    assert len(matches) == 1, f"Expected one {name}, got {len(matches)}"
    return matches[0]


def test_initialization_has_distinct_positions_then_new_measurement_and_pending(
    render_svg,
):
    campaign = shared_campaign()
    payload = plot_payload(campaign)
    svg = render_svg(payload)
    axis_ticks = ticks(svg)
    circles = measured_circles(svg)
    assert len(circles) == 4
    by_recipe = {
        index: next(node for node in circles if f"recipe {index}" in title(node))
        for index in range(4)
    }
    positions = [float(by_recipe[index].attrib["cx"]) for index in range(4)]
    assert positions == sorted(
        set(positions)
    ), "Seeds and later measurements need separate horizontal positions"
    for index, label in enumerate(["i1", "i2", "i3", "1"]):
        assert positions[index] == pytest.approx(axis_ticks[label], abs=0.1)
        assert by_recipe[index].attrib["data-observation-index"] == str(index + 1)
        assert by_recipe[index].attrib["data-axis-label"] == label
        phase = "initialization" if index < 3 else "new measurement"
        assert title(by_recipe[index]).startswith(f"{label} ({phase}): recipe {index}:")
    assert all("initial" in title(by_recipe[index]).lower() for index in range(3))
    assert "initial" not in title(by_recipe[3]).lower()

    predictions = [
        node for node in svg.iter("path") if "Awaiting measurement" in title(node)
    ]
    assert len(predictions) == 1
    pending = predictions[0]
    center = float(re.match(r"M\s+([-\d.]+)", pending.attrib["d"]).group(1))
    assert center == pytest.approx(axis_ticks["2"], abs=0.1)
    assert center > positions[-1]
    assert title(pending).startswith("Awaiting measurement predicted: 97;")
    assert "95% latent-function interval 92 to 100" in title(pending)
    assert payload["plot_counts"]["new_completed"] == 1
    assert payload["best_trace"][-1]["best"] == 91

    shade = by_marker(svg, "initialization-region")
    divider = by_marker(svg, "initialization-divider")
    left = float(shade.attrib["x"])
    edge = left + float(shade.attrib["width"])
    assert shade.tag == "rect" and divider.tag == "line"
    assert left < positions[0] < positions[2] < edge < positions[3]
    assert edge == pytest.approx(float(divider.attrib["x1"]), abs=0.1)
    assert divider.attrib["x1"] == divider.attrib["x2"]
    assert "initialization" in " ".join(svg.itertext()).lower()


def test_initialization_only_remains_separated_inside_shaded_region(render_svg):
    payload = plot_payload(shared_campaign(new_measurement=False, pending=False))
    svg = render_svg(payload)
    circles = measured_circles(svg)
    positions = [float(node.attrib["cx"]) for node in circles]
    assert len(positions) == len(set(positions)) == 3
    assert set(["i1", "i2", "i3"]).issubset(ticks(svg))
    shade = by_marker(svg, "initialization-region")
    right = float(shade.attrib["x"]) + float(shade.attrib["width"])
    assert max(positions) < right < float(svg.attrib["viewBox"].split()[2])
    assert payload["plot_counts"]["new_completed"] == 0


def test_legacy_plot_without_initialization_metadata_keeps_regular_axis(render_svg):
    payload = dict(
        config={},
        observations=[
            dict(procedure="legacy first", value=12),
            dict(procedure="legacy second", value=17),
        ],
        best_trace=[dict(index=1, best=12), dict(index=2, best=17)],
    )
    original = deepcopy(payload)
    svg = render_svg(payload)
    assert payload == original
    assert {"1", "2", "experiment count"}.issubset(ticks(svg))
    assert not any("data-initialization-region" in node.attrib for node in svg.iter())
    circles = [node for node in svg.iter("circle") if "legacy " in title(node)]
    assert len(circles) == 2
    assert float(circles[0].attrib["cx"]) < float(circles[1].attrib["cx"])
    assert not any((node.text or "").startswith("i1") for node in svg.iter("text"))


def test_excluded_measurement_is_a_marked_gap_and_later_step_is_not_renumbered(
    render_svg,
):
    data = shared_campaign(pending=True)
    data["observations"][-1].update(
        training_included=False,
        training_exclusion={"reason": "Incompatible quantification"},
    )
    later = deepcopy(data["observations"][-1])
    later.update(
        candidate_id="4",
        observation_id="later",
        physical_measurement_id="later",
        training_included=True,
        moc_wt_pct=80,
    )
    data["observations"].append(later)
    payload = plot_payload(data)
    svg = render_svg(payload)
    gaps = [
        node
        for node in svg.iter("path")
        if "data-excluded-measurement-index" in node.attrib
    ]
    assert len(gaps) == 1
    assert gaps[0].attrib["data-excluded-measurement-index"] == "4"
    assert gaps[0].attrib["data-axis-label"] == "1"
    assert "outcome not plotted" in title(gaps[0])
    assert "Incompatible quantification" in title(gaps[0])
    circles = measured_circles(svg)
    assert not any("recipe 3" in title(node) for node in circles)
    second = next(node for node in circles if "recipe 4" in title(node))
    assert second.attrib["data-axis-label"] == "2"
    assert float(second.attrib["cx"]) == pytest.approx(ticks(svg)["2"], abs=0.1)
    assert payload["plot_counts"]["new_completed"] == 2


def test_all_excluded_initialization_still_renders_occupied_positions(render_svg):
    data = shared_campaign(new_measurement=False, pending=False)
    data["config"]["bounds"] = [None, None]
    for row in data["observations"]:
        row["training_included"] = False
    svg = render_svg(plot_payload(data))
    assert not measured_circles(svg)
    assert [
        node.attrib["data-axis-label"]
        for node in svg.iter("path")
        if "data-excluded-measurement-index" in node.attrib
    ] == ["i1", "i2", "i3"]
    assert {"i1", "i2", "i3"}.issubset(ticks(svg))
    by_marker(svg, "initialization-divider")
