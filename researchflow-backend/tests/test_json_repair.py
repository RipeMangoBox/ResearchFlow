"""Test _repair_truncated_json with realistic truncated inputs."""
import sys
sys.path.insert(0, ".")

from backend.services.analysis_steps import _parse_json_safe, _repair_truncated_json

# Case 1: truncated mid-string in confidence_notes array
test1 = '{"problem_summary": "MLLMs lack ego-grounding ability", "method_summary": "Built MyEgo dataset: 541 videos", "evidence_summary": "GPT-5 only 46% accuracy", "core_intuition": "Need persistent identity anchor", "changed_slots": ["evaluation_framework"], "confidence_notes": [{"claim": "method works", "confidence": 0.8, "basis": "exp", "reasoning": "multi-model eval but limited to self-built dataset directi'

# Case 2: truncated after a complete top-level value
test2 = '{"problem_summary": "Surgery VQA lacks temporal modeling", "method_summary": "Proposed SurgTEMP framework with text-guided memory pyramid", "evidence_summary": "Outperforms baselines on CholeVidQA-32K'

# Case 3: truncated with Chinese text
test3 = '{\n  "problem_summary": "\u8179\u8154\u955c\u80c6\u56ca\u5207\u9664\u672f\u7b49\u5916\u79d1\u624b\u672f\u89c6\u9891\u5177\u6709\u9ad8\u5ea6\u590d\u6742\u6027",\n  "method_summary": "SurgTEMP\u6846\u67b6\u548cCholeVidQA-32K\u6570\u636e\u96c6",\n  "delta_card": {"baseline": "generic VLMs", "mechanism": "text-guided memory pyramid"'

for i, (name, test) in enumerate([(1, test1), (2, test2), (3, test3)], 1):
    print(f"\n=== Test {name} (len={len(test)}) ===")

    # Direct repair
    repair = _repair_truncated_json(test)
    if repair:
        print(f"  _repair: OK, {len(repair)} keys: {list(repair.keys())}")
    else:
        print(f"  _repair: FAILED")

    # Full pipeline
    result = _parse_json_safe(test)
    print(f"  _parse_json_safe: {len(result)} keys: {list(result.keys())}")
    for k in ["problem_summary", "method_summary", "core_intuition"]:
        v = result.get(k)
        if v:
            print(f"    {k}: {v[:50]}...")

print("\n=== ALL TESTS DONE ===")
