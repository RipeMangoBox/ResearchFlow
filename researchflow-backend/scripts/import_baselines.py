"""Import core baseline papers that are referenced by the existing 24 but missing from KB.

These are the most-cited baselines in experiment tables across the Motion Generation papers.
Each is imported as a new Paper row with arxiv_id for later enrichment.
"""

import psycopg2
import uuid

DB_URL = "postgresql://hzh@localhost:5432/researchflow"

BASELINES = [
    {
        "title": "MDM: Human Motion Diffusion Model",
        "arxiv_id": "2209.14916",
        "venue": "ICLR",
        "year": 2023,
        "category": "Motion_Generation_Text_Speech_Music_Driven",
    },
    {
        "title": "MLD: Execute Your Commands via Motion Diffusion in Latent Space",
        "arxiv_id": "2212.04048",
        "venue": "CVPR",
        "year": 2023,
        "category": "Motion_Generation_Text_Speech_Music_Driven",
    },
    {
        "title": "MotionDiffuse: Text-Driven Human Motion Generation with Diffusion Model",
        "arxiv_id": "2208.15001",
        "venue": "TPAMI",
        "year": 2024,
        "category": "Motion_Generation_Text_Speech_Music_Driven",
    },
    {
        "title": "TEACH: Temporal Action Composition for 3D Humans",
        "arxiv_id": "2209.04066",
        "venue": "3DV",
        "year": 2022,
        "category": "Motion_Generation_Text_Speech_Music_Driven",
    },
    {
        "title": "TEMOS: Generating diverse human motions from textual descriptions",
        "arxiv_id": "2204.14109",
        "venue": "ECCV",
        "year": 2022,
        "category": "Motion_Generation_Text_Speech_Music_Driven",
    },
    {
        "title": "ReMoDiffuse: Retrieval-Augmented Motion Diffusion Model",
        "arxiv_id": "2304.01116",
        "venue": "ICCV",
        "year": 2023,
        "category": "Motion_Generation_Text_Speech_Music_Driven",
    },
    {
        "title": "PhysDiff: Physics-Guided Human Motion Diffusion Model",
        "arxiv_id": "2212.02500",
        "venue": "ICCV",
        "year": 2023,
        "category": "Human_Object_Interaction",
    },
    {
        "title": "OMOMO: Object Motion Guided Human Motion Synthesis",
        "arxiv_id": "2309.16237",
        "venue": "SIGGRAPH Asia",
        "year": 2023,
        "category": "Human_Object_Interaction",
    },
    {
        "title": "TM2T: Stochastic and Tokenized Modeling for the Reciprocal Generation of 3D Human Motions and Texts",
        "arxiv_id": "2207.01696",
        "venue": "ECCV",
        "year": 2022,
        "category": "Motion_Generation_Text_Speech_Music_Driven",
    },
]


def import_baselines():
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    cur = conn.cursor()

    print("=== Importing baseline papers ===\n")

    imported = 0
    skipped = 0

    for b in BASELINES:
        # Check if already exists (by arxiv_id or title)
        cur.execute(
            "SELECT id FROM papers WHERE arxiv_id = %s OR title ILIKE %s LIMIT 1",
            (b["arxiv_id"], f"%{b['title'][:60]}%"),
        )
        existing = cur.fetchone()
        if existing:
            print(f"  SKIP (exists): {b['title'][:60]}...")
            skipped += 1
            continue

        paper_id = uuid.uuid4()
        title_sanitized = (
            b["title"]
            .replace(":", " -")
            .replace("/", "-")
            .replace("?", "")
            .strip()
        )

        cur.execute("""
            INSERT INTO papers (id, title, title_sanitized, arxiv_id, venue, year,
                               category, state, importance, tags,
                               paper_link, collected_at, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'wait', 'A', %s, %s, now(), now(), now())
        """, (
            str(paper_id),
            b["title"],
            title_sanitized,
            b["arxiv_id"],
            b["venue"],
            b["year"],
            b["category"],
            [b["category"], f"task/text-to-motion", "baseline"],
            f"https://arxiv.org/abs/{b['arxiv_id']}",
        ))

        print(f"  IMPORTED: {b['title'][:60]}... (arxiv:{b['arxiv_id']})")
        imported += 1

    print(f"\nImported: {imported}, Skipped: {skipped}")

    cur.execute("SELECT count(*) FROM papers")
    print(f"Total papers in DB: {cur.fetchone()[0]}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    import_baselines()
