"""Canonicalize taxonomy_node names so dataset/task nodes don't fragment.

Without this, every paper's L3 facet extraction creates idiosyncratic
variants ("CIFAR-10", "CIFAR-10 OOD detection", "CIFAR-10_NIID-1_(α=0.1)",
"CIFAR10") and 89% of dataset nodes end up with exactly one paper —
defeating the cross-paper graph entirely.

Strategy: collapse trailing parentheticals + setting suffixes + minor
spelling drift to a *canonical* name:

    "CIFAR-10 OOD detection"            → "CIFAR-10"
    "CIFAR-10_NIID-1_(α=0.1, α=0.01)"   → "CIFAR-10"
    "CIFAR10"                            → "CIFAR-10"
    "ImageNet-1K Classification"        → "ImageNet-1K"
    "ImageNet-1K / MobileNetV2 …"       → "ImageNet-1K"
    "BraTS2018 balanced missing"        → "BraTS2018"
    "Breakfast 10-task incremental (ASFormer)" → "Breakfast"

Used in two places:
  * vault_export_v6._export_papers / _export_datasets — render canonical names.
  * scripts/merge_facet_variants.py — one-shot SQL to merge variants in DB.
"""

from __future__ import annotations

import re

# ── Canonical regex rules ──────────────────────────────────────────────
# Order matters — first match wins. Tested with `_test_canonicalize`.

_RULES: list[tuple[str, str]] = [
    # CIFAR — order CIFAR-100 first so 10-prefix rule doesn't swallow it.
    # We can't use \b after digits because `_` is a word char.
    (r"^CIFAR[\s\-_]?100(?!\d).*",                           "CIFAR-100"),
    (r"^CIFAR[\s\-_]?10(?!\d).*",                            "CIFAR-10"),
    # ImageNet family
    (r"^Tiny[\s\-_]?ImageNet\b.*",                           "Tiny-ImageNet"),
    (r"^ImageNet[\s\-_]?1[Kk]\b.*",                          "ImageNet-1K"),
    (r"^ImageNet[\s\-_]?200\b.*",                            "ImageNet-200"),
    (r"^ImageNet[\s\-_]?22[Kk]\b.*",                         "ImageNet-22K"),
    (r"^ImageNet[\s\-_]?21[Kk]\b.*",                         "ImageNet-21K"),
    (r"^ImageNet16[\s\-_]?120\b.*",                          "ImageNet16-120"),
    # Bare "ImageNet" with no size suffix → ImageNet-1K (ILSVRC-1000).
    # Papers using the 22K/21K variants always specify the size; bare
    # "ImageNet" almost always means the 1000-class classification subset.
    # Without this collapse, D__ImageNet and D__ImageNet-1K split the same
    # dataset across two graph nodes.
    (r"^ImageNet\b.*",                                       "ImageNet-1K"),
    # COCO / VOC / Pascal
    (r".*\bCOCO[\s\-_]?Instance.*",                          "COCO"),
    (r".*\bCOCO[\s\-_]?Detection.*",                         "COCO"),
    (r".*\bCOCO[\s\-_]?Caption.*",                           "MSCOCO Captions"),
    (r".*\bMSCOCO[\s\-_]?Image[\s\-_]?Captioning.*",         "MSCOCO Captions"),
    (r".*\bMSCOCO[\s\-_]?Image[\s\-_]?Text[\s\-_]?Retrieval.*", "MSCOCO Retrieval"),
    (r"^MSCOCO\b.*|^MS[\s\-_]?COCO\b.*|^COCO\b.*",           "MS-COCO"),
    (r"^Pascal[\s\-_]?VOC\b.*",                              "Pascal VOC"),
    # NAS-Bench
    (r"^NAS[\s\-_]?Bench[\s\-_]?201\b.*",                    "NAS-Bench-201"),
    (r"^NAS[\s\-_]?Bench[\s\-_]?101\b.*",                    "NAS-Bench-101"),
    (r"^NAS[\s\-_]?Bench[\s\-_]?Macro\b.*",                  "NAS-Bench-Macro"),
    # Medical
    (r"^BraTS\d{2,4}\b.*",                                   None),  # keep year+name, see below
    # LLaVA family
    (r"^LLaVA[\s\-_]?Bench\b.*",                             "LLaVA-Bench"),
    (r"^LLaVA[\s\-_]?Wilder\b.*",                            "LLaVA-Wilder"),
    (r"^LLaVA[\s\-_]?Critic[\s\-_]?Train.*",                 "LLaVA-Critic-Train"),
    # Breakfast / 50Salads / common video
    (r"^Breakfast\b.*",                                      "Breakfast"),
    (r"^50Salads\b.*",                                       "50Salads"),
    (r"^GTEA\b.*",                                           "GTEA"),
    # WildVision / SEED / MMHal / MM-Vet
    (r"^WildVision[\s\-_]?Bench\b.*",                        "WildVision-Bench"),
    (r"^SEED[\s\-_]?Bench\b.*",                              "SEED-Bench"),
    (r"^MMHal[\s\-_]?Bench\b.*",                             "MMHal-Bench"),
    (r"^MM[\s\-_]?Vet\b.*",                                  "MM-Vet"),
    # Argoverse
    (r"^Argoverse\b.*",                                      "Argoverse"),
    (r"^AV2\b.*",                                            "AV2"),
    # PEDES
    (r"^RSTPReid\b.*",                                       "RSTPReid"),
    (r"^CUHK[\s\-_]?PEDES\b.*",                              "CUHK-PEDES"),
    (r"^ICFG[\s\-_]?PEDES\b.*",                              "ICFG-PEDES"),
    # SWIG-HOI / HICO-DET
    (r"^SWIG[\s\-_]?HOI\b.*",                                "SWIG-HOI"),
    (r"^HICO[\s\-_]?DET\b.*",                                "HICO-DET"),
    # Time series — too generic to canonicalize, keep as-is below
    # Reasoning benchmarks
    (r"^ARC[\s\-_]?Challenge\b.*|^ARC[\s\-_]?C\b.*",         "ARC-Challenge"),
    (r"^ARC[\s\-_]?Easy\b.*|^ARC[\s\-_]?E\b.*",              "ARC-Easy"),
    (r"^BoolQ\b.*",                                          "BoolQ"),
    (r"^AIME\s*\d{2,4}\b.*",                                 None),  # keep year
    # Aggregate / generic suffixes — strip them
]

# Suffixes to strip after applying main rules (apply in second pass)
_SUFFIX_STRIP = re.compile(
    r"\s*[\(\[\{].*$"               # trailing (...) [...]
    r"|\s*/.*$"                      # trailing /-clauses (e.g. "/MobileNetV2 …")
    r"|\s*[—–-]\s*aggregate.*$"
    r"|\s*[Ss]mall[\s\-_]?scale.*$"
    r"|\s*[Aa]verage.*$"
    r"|\s*comparison.*$"
    r"|\s*[A-Za-z]+[\s\-_]?dataset[\s\-_]?condensation.*$"
    r"|\s+(balanced|imbalanced|missing|complete|partial|train|val|test|dev)\b.*$"
    r"|\s+(classification|detection|segmentation|retrieval|matching|generation)\b.*$",
    re.IGNORECASE,
)


# ── Task-specific canonicalization ─────────────────────────────────────
# A "task" should describe the ACTIVITY, not the dataset+activity combo.
# `canonicalize_task_name` strips the leading dataset/benchmark and maps to
# a small set of canonical task labels.

# Common dataset/benchmark prefixes to strip from task names. Order matters
# (longest first) so ImageNet-1K is stripped before ImageNet.
_TASK_PREFIX_STRIP = [
    r"CIFAR[\s\-_]?(?:10|100)\b",
    r"ImageNet[\s\-_]?1[Kk]\b",
    r"ImageNet[\s\-_]?22[Kk]\b",
    r"ImageNet[\s\-_]?200\b",
    r"Tiny[\s\-_]?ImageNet\b",
    r"ImageNet\b",
    r"COCO\b", r"MSCOCO\b", r"MS[\s\-_]?COCO\b",
    r"Pascal[\s\-_]?VOC\b",
    r"BraTS\d{0,4}\b",
    r"NAS[\s\-_]?Bench[\s\-_]?\d+\b",
    r"NYUv\d\b",
    r"Cityscapes\b",
    r"ScanNet\b",
    r"VOC\d{4}\b",
    r"Kinetics(?:[\s\-_]?\d+)?\b",
    r"AV2\b",
]

# Canonical task labels — substring lookup
_TASK_CANONICAL = [
    (r"\bOOD[\s\-_]?detection\b|\bout[\s\-_]?of[\s\-_]?distribution\b", "OOD Detection"),
    (r"\bclassification\b",                                              "Classification"),
    (r"\b(?:object[\s\-_]?)?detection\b",                                "Object Detection"),
    (r"\bsemantic[\s\-_]?segmentation\b",                                "Semantic Segmentation"),
    (r"\binstance[\s\-_]?segmentation\b",                                "Instance Segmentation"),
    (r"\bsegmentation\b",                                                "Segmentation"),
    (r"\b(?:image[\s\-_]?)?retrieval\b",                                 "Retrieval"),
    (r"\b(?:cross[\s\-_]?modal|multi[\s\-_]?modal)[\s\-_]?matching\b",   "Cross-Modal Matching"),
    (r"\bmatching\b",                                                    "Matching"),
    # Generation: keep modality split — collapsing all into "Generation"
    # creates a 61-paper hub. Image/Video/Text/Audio Generation are
    # distinct research areas.
    (r"\bimage[\s\-_]?generation\b|\bimg[\s\-_]?gen\b|\btext[\s\-_]?to[\s\-_]?image\b|\bT2I\b", "Image Generation"),
    (r"\bvideo[\s\-_]?generation\b|\btext[\s\-_]?to[\s\-_]?video\b|\bT2V\b",  "Video Generation"),
    (r"\btext[\s\-_]?generation\b",                                       "Text Generation"),
    (r"\baudio[\s\-_]?generation\b|\bsound[\s\-_]?generation\b",          "Audio Generation"),
    (r"\bcode[\s\-_]?generation\b",                                       "Code Generation"),
    (r"\bmotion[\s\-_]?generation\b",                                     "Motion Generation"),
    (r"\bgeneration\b",                                                   "Generation"),
    (r"\b(?:reinforcement[\s\-_]?learning|RL)\b",                        "Reinforcement Learning"),
    (r"\b(?:domain[\s\-_]?adaptation|DA)\b",                             "Domain Adaptation"),
    (r"\b(?:federated[\s\-_]?learning|FL)\b",                            "Federated Learning"),
    (r"\bcontinual[\s\-_]?learning\b|\blifelong[\s\-_]?learning\b",      "Continual Learning"),
    (r"\bunlearning\b",                                                  "Machine Unlearning"),
    (r"\b(?:NAS|neural[\s\-_]?architecture[\s\-_]?search)\b",            "Neural Architecture Search"),
    (r"\bcompression\b",                                                 "Compression"),
    (r"\b(?:VQA|visual[\s\-_]?question[\s\-_]?answering)\b",             "Visual Question Answering"),
    (r"\bcaption(?:ing)?\b",                                             "Captioning"),
    (r"\b(?:video|action)[\s\-_]?(?:recognition|understanding)\b",       "Video Understanding"),
    (r"\bspeech[\s\-_]?(?:recognition|synthesis)\b",                     "Speech Processing"),
    (r"\bevaluat(?:ion|ing|or)\b|\bbench[\s\-_]?mark\b",                 "Benchmark / Evaluation"),
    (r"\bagent\b",                                                       "Agent"),
    # Reasoning: keep specific variants distinct
    (r"\bmath(?:ematical)?[\s\-_]?reasoning\b",                          "Math Reasoning"),
    (r"\bvisual[\s\-_]?reasoning\b",                                     "Visual Reasoning"),
    (r"\blogical[\s\-_]?reasoning\b|\bdeductive[\s\-_]?reasoning\b",     "Logical Reasoning"),
    (r"\bcommonsense[\s\-_]?reasoning\b",                                "Commonsense Reasoning"),
    (r"\breasoning\b",                                                   "Reasoning"),
]


def canonicalize_task_name(raw: str) -> str:
    """Tasks describe activities, not datasets+activities. Strip leading
    dataset name and map to a small canonical label set.

    Examples:
        "CIFAR-10 OOD detection"           → "OOD Detection"
        "ImageNet-1K Classification"       → "Classification"
        "COCO Instance Segmentation (Swin-B)" → "Instance Segmentation"
        "Open-vocabulary 3D detection"     → "Object Detection"
        "Reinforcement Learning agent"     → "Reinforcement Learning"
    """
    if not raw:
        return raw
    s = re.sub(r"\s+", " ", raw).strip()

    # 1) Strip leading dataset/benchmark name
    for pat in _TASK_PREFIX_STRIP:
        s = re.sub(rf"^{pat}\s*[/:\-_]?\s*", "", s, flags=re.IGNORECASE)

    # 2) Map to canonical activity label
    for pat, canon in _TASK_CANONICAL:
        if re.search(pat, s, flags=re.IGNORECASE):
            return canon

    # 3) Fallback: use the dataset normalizer to strip parentheticals
    return canonicalize_facet_name(s, "task")


def canonicalize_facet_name(raw: str, dimension: str = "dataset") -> str:
    """Return the canonical form for a dataset/task node name.

    `dimension` is informational — currently same logic for dataset and task.
    Returns a stripped, single-spaced string. Never returns empty.
    """
    if not raw:
        return raw
    s = re.sub(r"\s+", " ", raw).strip()

    # 1. Apply explicit rules
    for pattern, replacement in _RULES:
        if re.match(pattern, s, flags=re.IGNORECASE):
            if replacement is not None:
                return replacement
            # None means "keep s but still apply suffix strip"
            break

    # 2. Strip trailing parentheticals / suffix decorations
    s2 = _SUFFIX_STRIP.sub("", s).strip()
    return s2 or s


# ── Self-test ──────────────────────────────────────────────────────────

def _test_canonicalize_task():
    cases = [
        ("CIFAR-10 OOD detection",                  "OOD Detection"),
        ("ImageNet-1K Classification",              "Classification"),
        ("COCO Instance Segmentation (Swin-B)",     "Instance Segmentation"),
        ("Pascal VOC Detection (Swin-L)",           "Object Detection"),
        ("Reinforcement Learning agent",            "Reinforcement Learning"),
        ("Open-vocabulary 3D detection",            "Object Detection"),
        ("Federated Learning under non-IID",        "Federated Learning"),
    ]
    fails = []
    for raw, want in cases:
        got = canonicalize_task_name(raw)
        if got != want:
            fails.append((raw, want, got))
    if fails:
        for f in fails:
            print(f"TASK FAIL {f[0]!r} → expected {f[1]!r} got {f[2]!r}")
        return False
    print(f"all {len(cases)} task cases pass")
    return True


def _test_canonicalize():
    cases = [
        ("CIFAR-10",                                "CIFAR-10"),
        ("CIFAR10",                                 "CIFAR-10"),
        ("CIFAR-10 OOD detection",                  "CIFAR-10"),
        ("CIFAR-10_NIID-1_(α=0.1, α=0.01)",         "CIFAR-10"),
        ("CIFAR-10-CINIC-10 average",               "CIFAR-10"),
        ("CIFAR-100 OOD detection",                 "CIFAR-100"),
        ("ImageNet-1K Classification",              "ImageNet-1K"),
        ("ImageNet-1K / MobileNetV2 search space (~450M FLOPs)", "ImageNet-1K"),
        ("Tiny-ImageNet",                           "Tiny-ImageNet"),
        ("Breakfast (TAS dataset condensation)",    "Breakfast"),
        ("Breakfast 10-task incremental (ASFormer)","Breakfast"),
        ("50Salads",                                "50Salads"),
        ("BraTS2018 balanced missing",              "BraTS2018"),
        ("BraTS2020 imbalanced missing",            "BraTS2020"),
        ("LLaVA-Wilder (L-Wilder)",                 "LLaVA-Wilder"),
        ("LLaVA-Bench (LLaVA-W)",                   "LLaVA-Bench"),
        ("MMHal-Bench",                             "MMHal-Bench"),
        ("Pascal VOC Detection (Swin-L)",           "Pascal VOC"),
        ("NAS-Bench-201",                           "NAS-Bench-201"),
        ("NAS-Bench-201 / ImageNet16-120",          "NAS-Bench-201"),
        ("ImageNet16-120",                          "ImageNet16-120"),
    ]
    fails = []
    for raw, want in cases:
        got = canonicalize_facet_name(raw)
        if got != want:
            fails.append((raw, want, got))
    if fails:
        for f in fails:
            print(f"FAIL {f[0]!r} → expected {f[1]!r} got {f[2]!r}")
        return False
    print(f"all {len(cases)} cases pass")
    return True


if __name__ == "__main__":
    _test_canonicalize()
    _test_canonicalize_task()
