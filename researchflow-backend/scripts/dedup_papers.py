"""Deduplicate papers: merge duplicate pairs, keep the one with arxiv_id.

Known duplicates in the Motion Generation dataset:
1. InterMask...Modeling vs InterMask...Modelling (ICLR 2025)
2. MotionGPT...Foreign Language vs ...a Foreign Language (NeurIPS 2023)
3. DART... vs DartControl... (ICLR 2025)
4. LaMP: Language-Motion... vs Language-Motion Pretraining... (ICLR 2025)

Strategy: keep the row with arxiv_id, delete the other, update any FK references.
"""

import psycopg2

DB_URL = "postgresql://hzh@localhost:5432/researchflow"


DUPLICATE_PAIRS = [
    # (title_fragment_to_KEEP (has arxiv_id), title_fragment_to_DELETE)
    ("InterMask: 3D Human Interaction Generation via Collaborative Masked Modelling",
     "InterMask: 3D Human Interaction Generation via Collaborative Masked Modeling"),
    ("MotionGPT: Human Motion as Foreign Language",
     "MotionGPT: Human Motion as a Foreign Language"),
    ("DART: A Diffusion-Based Autoregressive Motion Model",
     "DartControl: A Diffusion-Based Autoregressive Motion Model"),
    ("Language-Motion Pretraining for Motion Generation,Retrieval,and Captioning",
     "LaMP: Language-Motion Pretraining for Motion Generation, Retrieval, and Captioning"),
]


def dedup():
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    cur = conn.cursor()

    print("=== Deduplicating papers ===\n")

    for keep_frag, del_frag in DUPLICATE_PAIRS:
        # Find IDs
        cur.execute("SELECT id, title, arxiv_id FROM papers WHERE title LIKE %s",
                     (f"%{keep_frag}%",))
        keep_rows = cur.fetchall()

        cur.execute("SELECT id, title, arxiv_id FROM papers WHERE title LIKE %s",
                     (f"%{del_frag}%",))
        del_rows = cur.fetchall()

        if not keep_rows or not del_rows:
            print(f"  SKIP: pair not found — keep='{keep_frag[:40]}...' del='{del_frag[:40]}...'")
            continue

        keep_id = keep_rows[0][0]
        del_id = del_rows[0][0]

        if keep_id == del_id:
            print(f"  SKIP: same row — '{keep_frag[:50]}...'")
            continue

        keep_arxiv = keep_rows[0][2] or "none"
        del_arxiv = del_rows[0][2] or "none"

        print(f"  MERGE: keep={keep_arxiv} ({str(keep_id)[:8]}), delete={del_arxiv} ({str(del_id)[:8]})")
        print(f"    keep: {keep_rows[0][1][:60]}")
        print(f"    del:  {del_rows[0][1][:60]}")

        # If the delete row has arxiv_id but keep doesn't, copy it over
        if del_arxiv != "none" and keep_arxiv == "none":
            cur.execute("UPDATE papers SET arxiv_id = %s WHERE id = %s", (del_rows[0][2], keep_id))
            print(f"    copied arxiv_id {del_arxiv} to keep row")

        # Copy abstract/authors if keep row is missing them
        cur.execute("""
            UPDATE papers SET
                abstract = COALESCE(papers.abstract, src.abstract),
                authors = COALESCE(papers.authors, src.authors)
            FROM papers src
            WHERE papers.id = %s AND src.id = %s
        """, (keep_id, del_id))

        # Update FK references from deleted → kept
        for table, col in [
            ("paper_analyses", "paper_id"),
            ("delta_cards", "paper_id"),
            ("evidence_units", "paper_id"),
        ]:
            try:
                cur.execute(f"UPDATE {table} SET {col} = %s WHERE {col} = %s", (keep_id, del_id))
            except Exception:
                pass

        # Delete the duplicate
        cur.execute("DELETE FROM papers WHERE id = %s", (del_id,))
        print(f"    deleted {str(del_id)[:8]}")

    # Final count
    cur.execute("SELECT count(*) FROM papers WHERE state = 'checked'")
    print(f"\nRemaining checked papers: {cur.fetchone()[0]}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    dedup()
