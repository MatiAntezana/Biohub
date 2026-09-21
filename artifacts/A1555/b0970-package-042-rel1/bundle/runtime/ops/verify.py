"""Local static/CPU verification. Full SCIP/runtime tests remain mandatory on UCBV."""
import argparse
import ast
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.cli_bootstrap import configure_cli_imports
configure_cli_imports(ROOT)
sys.path.insert(0, str(ROOT / "tests"))


def test_outcome(result, expected, full):
    return {"tests_passed": (result.wasSuccessful() and not result.skipped
            and result.testsRun == expected and expected > 0),
        "tests_run": result.testsRun, "tests_expected": expected, "skipped": len(result.skipped),
        "failures": len(result.failures), "errors": len(result.errors),
        "suite": "full_discovery" if full else "local_cpu"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true", help="Include real SCIP solver and tracksdata/GEFF/Zarr import tests")
    p.add_argument("--report", type=Path, help="Write a structured test receipt; never skip a failing test")
    args = p.parse_args()
    cpu_runtime = None
    if os.environ.get("BIOHUB_RESOURCE_PROFILE") == "cpu_evaluation_v1":
        from ops.biohub import configure_cpu_worker, verify_cpu_runtime, CPU_THREADS
        configure_cpu_worker("tests")
        import torch
        torch.set_num_threads(CPU_THREADS)
        torch.set_num_interop_threads(1)
        cpu_runtime = {"before": verify_cpu_runtime(torch)}
    for directory in ("core", "ops", "tests", "vendor"):
        for path in (ROOT / directory).rglob("*.py"):
            ast.parse(path.read_text(), filename=str(path))
    for path in (ROOT / "configs").glob("*.json"):
        json.loads(path.read_text())
    for path in list((ROOT / "legacy").glob("*.ipynb")) + list((ROOT / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text())
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"{path}:cell{i}")
    import test_mechanisms as mechanisms
    import test_protocol as protocol
    import test_cli_imports as cli_imports
    import test_replay as replay
    import test_quantization as quantization
    import test_node_provenance as node_provenance
    import test_mitosis_diagnostic as mitosis_diagnostic
    import test_mitosis_learning as mitosis_learning
    import test_mitosis_probe as mitosis_probe
    import test_mitosis_proposals as mitosis_proposals
    import test_mitosis_counterfactual as mitosis_counterfactual
    import test_mitosis_selector as mitosis_selector
    import test_repair_ranking as repair_ranking
    import test_event_ranking as event_ranking
    import test_coverage_audit as coverage_audit
    import test_stratified_ranking_015 as stratified_ranking_015
    import test_temporal_association_016 as temporal_016
    import test_temporal_association_017 as temporal_017
    import test_temporal_association_018 as temporal_018
    import test_temporal_association_019 as temporal_019
    import test_temporal_association_020 as temporal_020
    import test_temporal_association_021 as temporal_021
    import test_temporal_association_022 as temporal_022
    import test_temporal_association_023 as temporal_023
    import test_temporal_association_024 as temporal_024
    import test_temporal_association_025 as temporal_025
    import test_temporal_association_026 as temporal_026
    import test_temporal_association_027 as temporal_027
    import test_temporal_association_028 as temporal_028
    import test_temporal_association_029 as temporal_029
    import test_temporal_association_030 as temporal_030
    import test_temporal_association_031 as temporal_031
    import test_temporal_association_032 as temporal_032
    import test_temporal_association_033 as temporal_033
    import test_temporal_association_034 as temporal_034
    import test_temporal_association_035 as temporal_035
    import test_temporal_association_036 as temporal_036
    import test_temporal_association_037 as temporal_037
    import test_temporal_association_038 as temporal_038
    import test_temporal_association_039 as temporal_039
    import test_temporal_association_040 as temporal_040
    import test_temporal_association_041 as temporal_041
    import test_temporal_association_042 as temporal_042
    suite = unittest.TestSuite()
    for cls in (mechanisms.CandidateTests, mechanisms.ImageEvidenceTests, mechanisms.MitosisTests,
                protocol.ProtocolTests, cli_imports.CliImportTests, replay.ReplayTests,
                quantization.QuantizationTests, node_provenance.NodeProvenanceTests,
                mitosis_diagnostic.MitosisDiagnosticTests, mitosis_learning.MitosisLearningTests,
                mitosis_probe.MitosisProbeTests, mitosis_proposals.MitosisProposalsTests,
                mitosis_counterfactual.MitosisCounterfactualTests,
                mitosis_counterfactual.WorkerResourceTests,
                mitosis_selector.AssociationSelectorTests, mitosis_selector.SelectorWorkflowTests,
                repair_ranking.RepairPolicyTests, repair_ranking.RepairWorkflowTests,
                repair_ranking.RetrievalTests,
                event_ranking.EventPolicyTests, event_ranking.EventLearningTests,
                event_ranking.EventDiagnosticTests, event_ranking.EventWorkflowTests,
                coverage_audit.NoveltyAuditTests, coverage_audit.CoverageGeometryTests,
                coverage_audit.CoverageWorkflowTests,
                stratified_ranking_015.StratifiedAssemblyTests,
                stratified_ranking_015.StratifiedFittingTests,
                stratified_ranking_015.StratifiedWorkflowTests,
                temporal_016.TemporalAssociation016Tests,
                temporal_017.TemporalAssociation017Tests,
                temporal_018.TemporalAssociation018Tests,
                temporal_019.TemporalAssociation019Tests,
                temporal_020.TemporalAssociation020Tests,
                temporal_021.TemporalAssociation021Tests,
                temporal_022.TemporalAssociation022Tests,
                temporal_023.TemporalAssociation023Tests,
                temporal_024.TemporalAssociation024Tests,
                temporal_025.TemporalAssociation025Tests,
                temporal_026.TemporalAssociation026Tests,
                temporal_027.TemporalAssociation027Tests,
                temporal_028.TemporalAssociation028Tests,
                temporal_029.TemporalAssociation029Tests,
                temporal_030.TemporalAssociation030Tests,
                temporal_031.TemporalAssociation031Tests,
                temporal_032.TemporalAssociation032Tests,
                temporal_033.TemporalAssociation033Tests,
                temporal_034.TemporalAssociation034Tests,
                temporal_035.TemporalAssociation035Tests,
                temporal_036.TemporalAssociation036Tests,
                temporal_037.TemporalAssociation037Tests,
                temporal_038.TemporalAssociation038Tests,
                temporal_039.TemporalAssociation039Tests,
                temporal_040.TemporalAssociation040Tests,
                temporal_041.TemporalAssociation041Tests,
                temporal_042.TemporalAssociation042Tests):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    if args.full:
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    expected = suite.countTestCases()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    outcome = test_outcome(result, expected, args.full)
    if cpu_runtime is not None:
        cpu_runtime["after"] = verify_cpu_runtime(torch)
        outcome["cpu_runtime"] = cpu_runtime
    if args.report:
        from core.artifacts import dump
        dump(args.report, outcome)
    print("STATIC_PARSE=PASS; SCIP_TESTS=" + ("INCLUDED" if args.full else "DEFERRED_TO_REMOTE_FULL_SUITE"))
    print("RUNTIME_IMPORT_TESTS=" + ("INCLUDED" if args.full else "DEFERRED_TO_REMOTE_FULL_SUITE"))
    if not outcome["tests_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
