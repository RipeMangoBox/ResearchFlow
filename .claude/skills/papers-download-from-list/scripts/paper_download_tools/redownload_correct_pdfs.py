from __future__ import annotations

from pathlib import Path

import requests

from check_paper_downloads import (  # type: ignore[import]
    LOG_PATH,
    PAPER_ROOT,
    match_entries_to_pdfs,
    parse_log,
    scan_pdfs,
)


# Titles whose PDF URLs we have just corrected in download_log_updated.txt
TARGET_TITLES = {
    "MoReact: Generating Reactive Motion from Textual Descriptions",
    "PriorMDM: Human Motion Diffusion as a Generative Prior",
    "OmniControl: Control Any Joint at Any Time for Human Motion Generation",
    "UNIMASKM: A Unified Masked Autoencoder with Patchified Skeletons for Motion Synthesis",
    "TMR: Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis",
    "GestureDiffuCLIP: Gesture Diffusion Model with CLIP Latents",
    "LGTM: Local-to-Global Text-Driven Human Motion Diffusion Models",
    "ROG: Guiding Human-Object Interactions with Rich Geometry and Relations",
    "Phys-Reach-Grasp: Learning Physics-Based Full-Body Human Reaching and Grasping from Brief Walking References",
    "F-HOI: Toward Fine-grained Semantic-Aligned 3D Human-Object Interactions",
    "Enhancing 3D Human Motion Prediction with Gaze-informed Affordance in 3D Scenes. Yu et al.",
    "LAMA: Locomotion-Action-Manipulation: Synthesizing Human-Scene Interactions in Complex 3D Environments",
}


def main() -> None:
    log_path = Path(LOG_PATH)
    paper_root = Path(PAPER_ROOT)

    print(f"Using log file: {log_path}")
    print(f"Using paper root: {paper_root}")

    entries = parse_log(log_path)
    pdf_paths = scan_pdfs(paper_root)
    match_entries_to_pdfs(entries, pdf_paths)

    session = requests.Session()

    for entry in entries:
        if entry.title not in TARGET_TITLES:
            continue

        if entry.local_pdf is None:
            print(f"[SKIP] No local PDF mapped for: {entry.title}")
            continue

        url = entry.pdf_url
        dest = entry.local_pdf
        dest.parent.mkdir(parents=True, exist_ok=True)

        print(f"[DOWNLOAD] {entry.title}")
        print(f"  URL: {url}")
        print(f"  -> {dest}")

        try:
            resp = session.get(url, timeout=120)
            resp.raise_for_status()
        except Exception as exc:
            print(f"  [ERROR] Failed to download {url}: {exc}")
            continue

        content_type = resp.headers.get("Content-Type", "").lower()
        if "application/pdf" not in content_type:
            print(f"  [WARN] Content-Type not PDF: {content_type}")

        try:
            dest.write_bytes(resp.content)
        except Exception as exc:
            print(f"  [ERROR] Failed to write to {dest}: {exc}")
            continue

        print("  [OK]")

    print("Done redownloading corrected PDFs.")


if __name__ == "__main__":
    main()

