"""Standalone schema patch: bring DB 007 to pipeline-ready state.

Adds missing columns to existing tables and creates all V6 tables
needed by the agent pipeline. Idempotent — safe to run multiple times.

Usage: python scripts/patch_schema_for_pipeline.py
"""

import psycopg2
import sys

DB_URL = "postgresql://hzh@localhost:5432/researchflow"


def run_patch():
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    cur = conn.cursor()

    print("=== Patching schema for pipeline ===\n")

    # ── 1. Papers table: add missing columns ──────────────────────
    print("[1/5] Patching papers table...")
    cur.execute("""
        ALTER TABLE papers ADD COLUMN IF NOT EXISTS acceptance_type VARCHAR(50);
        ALTER TABLE papers ADD COLUMN IF NOT EXISTS review_scores JSONB;
        ALTER TABLE papers ADD COLUMN IF NOT EXISTS cited_by_count SMALLINT;
        ALTER TABLE papers ADD COLUMN IF NOT EXISTS dblp_key VARCHAR(200);
        ALTER TABLE papers ADD COLUMN IF NOT EXISTS ring VARCHAR(20);
        ALTER TABLE papers ADD COLUMN IF NOT EXISTS current_delta_card_id UUID;
        ALTER TABLE papers ADD COLUMN IF NOT EXISTS role_in_kb VARCHAR(30);
        ALTER TABLE papers ADD COLUMN IF NOT EXISTS method_family VARCHAR(100);
        ALTER TABLE papers ADD COLUMN IF NOT EXISTS domain_id UUID;
        ALTER TABLE papers ADD COLUMN IF NOT EXISTS absorption_level SMALLINT DEFAULT 0;
    """)
    # Backfill method_family from mechanism_family if exists
    cur.execute("""
        UPDATE papers SET method_family = mechanism_family
        WHERE method_family IS NULL AND mechanism_family IS NOT NULL;
    """)
    print("  papers: 10 columns added")

    # ── 2. DeltaCard table: add missing columns ──────────────────
    print("[2/5] Patching delta_cards table...")
    cur.execute("""
        ALTER TABLE delta_cards ADD COLUMN IF NOT EXISTS key_equations JSONB;
        ALTER TABLE delta_cards ADD COLUMN IF NOT EXISTS key_figures JSONB;
        ALTER TABLE delta_cards ADD COLUMN IF NOT EXISTS publish_status VARCHAR(20) DEFAULT 'draft';
        ALTER TABLE delta_cards ADD COLUMN IF NOT EXISTS evidence_count SMALLINT;
        ALTER TABLE delta_cards ADD COLUMN IF NOT EXISTS changed_slots_json JSONB;
        ALTER TABLE delta_cards ADD COLUMN IF NOT EXISTS is_structural BOOLEAN;
        ALTER TABLE delta_cards ADD COLUMN IF NOT EXISTS same_family_paper_ids UUID[];
        ALTER TABLE delta_cards ADD COLUMN IF NOT EXISTS method_node_ids UUID[];
        ALTER TABLE delta_cards ADD COLUMN IF NOT EXISTS local_keyness_score FLOAT;
        ALTER TABLE delta_cards ADD COLUMN IF NOT EXISTS field_keyness_score FLOAT;
        ALTER TABLE delta_cards ADD COLUMN IF NOT EXISTS analysis_run_id UUID;
        ALTER TABLE delta_cards ADD COLUMN IF NOT EXISTS source_asset_hash VARCHAR(64);
        ALTER TABLE delta_cards ADD COLUMN IF NOT EXISTS model_run_id VARCHAR(50);
    """)
    print("  delta_cards: 13 columns added")

    # ── 3. Patch remaining tables ─────────────────────────────────
    print("[3/5] Patching remaining tables...")
    cur.execute("""
        ALTER TABLE paper_analyses ADD COLUMN IF NOT EXISTS extracted_figure_images JSONB;
        ALTER TABLE metadata_observations ADD COLUMN IF NOT EXISTS raw_payload_object_key TEXT;
        ALTER TABLE metadata_observations ALTER COLUMN observed_at SET DEFAULT now();
        ALTER TABLE model_runs ALTER COLUMN prompt_version TYPE VARCHAR(50);
        ALTER TABLE canonical_ideas ADD COLUMN IF NOT EXISTS domain VARCHAR(100);
        ALTER TABLE canonical_ideas ADD COLUMN IF NOT EXISTS contribution_count SMALLINT DEFAULT 0;
        ALTER TABLE canonical_ideas ADD COLUMN IF NOT EXISTS merged_into_id UUID;
        ALTER TABLE canonical_ideas ADD COLUMN IF NOT EXISTS aliases TEXT[];
        ALTER TABLE canonical_ideas ADD COLUMN IF NOT EXISTS tags TEXT[];
        ALTER TABLE contribution_to_canonical_idea ADD COLUMN IF NOT EXISTS confidence FLOAT;
        ALTER TABLE contribution_to_canonical_idea ADD COLUMN IF NOT EXISTS source VARCHAR(30) DEFAULT 'system_inferred';
        ALTER TABLE delta_card_lineage ADD COLUMN IF NOT EXISTS parent_delta_card_id UUID;
        ALTER TABLE delta_card_lineage ADD COLUMN IF NOT EXISTS child_delta_card_id UUID;
        ALTER TABLE delta_card_lineage ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'candidate';
        ALTER TABLE method_nodes ALTER COLUMN type SET DEFAULT 'mechanism_family';
    """)
    print("  Patched paper_analyses, metadata_observations, model_runs, canonical_ideas, contribution, delta_card_lineage, method_nodes")

    # ── 4. Create V6 tables ──────────────────────────────────────
    print("[4/5] Creating V6 tables...")

    cur.execute("""
    -- Domain management
    CREATE TABLE IF NOT EXISTS domain_specs (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        name VARCHAR(200) NOT NULL,
        description TEXT,
        status VARCHAR(20) DEFAULT 'active',
        scope_tasks TEXT[],
        scope_modalities TEXT[],
        scope_paradigms TEXT[],
        scope_seed_methods TEXT[],
        negative_scope TEXT[],
        seed_paper_ids UUID[],
        budget_metadata_candidates INT DEFAULT 500,
        budget_shallow_ingest INT DEFAULT 100,
        budget_deep_ingest INT DEFAULT 30,
        created_at TIMESTAMPTZ DEFAULT now(),
        updated_at TIMESTAMPTZ DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS domain_source_registry (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        domain_id UUID REFERENCES domain_specs(id),
        source_type VARCHAR(50) NOT NULL,
        source_config JSONB,
        enabled BOOLEAN DEFAULT true,
        last_synced_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS incremental_checkpoints (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        domain_id UUID REFERENCES domain_specs(id),
        checkpoint_type VARCHAR(50) NOT NULL,
        checkpoint_value JSONB NOT NULL,
        created_at TIMESTAMPTZ DEFAULT now()
    );

    -- Candidate pipeline
    CREATE TABLE IF NOT EXISTS paper_candidates (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        title TEXT NOT NULL,
        normalized_title TEXT,
        arxiv_id VARCHAR(30),
        doi VARCHAR(100),
        s2_paper_id VARCHAR(50),
        openalex_id VARCHAR(50),
        openreview_id VARCHAR(100),
        paper_link TEXT,
        discovered_from_paper_id UUID REFERENCES papers(id),
        discovered_from_domain_id UUID REFERENCES domain_specs(id),
        discovery_source VARCHAR(30) NOT NULL,
        discovery_reason VARCHAR(50),
        relation_hint VARCHAR(30),
        authors_json JSONB,
        venue VARCHAR(100),
        year SMALLINT,
        abstract TEXT,
        citation_count INT,
        code_url TEXT,
        metadata_json JSONB,
        status VARCHAR(25) DEFAULT 'discovered',
        absorption_level SMALLINT DEFAULT 0,
        ingested_paper_id UUID REFERENCES papers(id),
        reject_reason TEXT,
        duplicate_of_candidate_id UUID,
        duplicate_of_paper_id UUID REFERENCES papers(id),
        created_at TIMESTAMPTZ DEFAULT now(),
        updated_at TIMESTAMPTZ DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS ix_pc_arxiv_id ON paper_candidates(arxiv_id);
    CREATE INDEX IF NOT EXISTS ix_pc_doi ON paper_candidates(doi);
    CREATE INDEX IF NOT EXISTS ix_pc_status ON paper_candidates(status);

    CREATE TABLE IF NOT EXISTS candidate_scores (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        candidate_id UUID REFERENCES paper_candidates(id) ON DELETE CASCADE,
        score_type VARCHAR(30) NOT NULL,
        total FLOAT NOT NULL,
        raw_total FLOAT,
        breakdown JSONB,
        hard_caps JSONB,
        boosts JSONB,
        penalties JSONB,
        decision VARCHAR(30),
        created_at TIMESTAMPTZ DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS score_signals (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        entity_type VARCHAR(30) NOT NULL,
        entity_id UUID NOT NULL,
        signal_name VARCHAR(50) NOT NULL,
        signal_value FLOAT,
        signal_json JSONB,
        source VARCHAR(50),
        created_at TIMESTAMPTZ DEFAULT now()
    );

    -- Agent infrastructure
    CREATE TABLE IF NOT EXISTS agent_runs (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        paper_id UUID REFERENCES papers(id),
        candidate_id UUID REFERENCES paper_candidates(id),
        domain_id UUID REFERENCES domain_specs(id),
        agent_name VARCHAR(50) NOT NULL,
        phase VARCHAR(20) NOT NULL,
        status VARCHAR(15) DEFAULT 'running',
        model_name VARCHAR(50),
        prompt_version VARCHAR(20),
        input_token_count INT,
        output_token_count INT,
        cost_usd FLOAT,
        duration_ms INT,
        error_message TEXT,
        started_at TIMESTAMPTZ DEFAULT now(),
        completed_at TIMESTAMPTZ
    );
    CREATE INDEX IF NOT EXISTS ix_ar_paper ON agent_runs(paper_id);

    CREATE TABLE IF NOT EXISTS agent_blackboard_items (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        run_id UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
        paper_id UUID REFERENCES papers(id),
        candidate_id UUID REFERENCES paper_candidates(id),
        item_type VARCHAR(30) NOT NULL,
        value_json JSONB NOT NULL,
        confidence FLOAT,
        evidence_refs JSONB,
        producer_agent VARCHAR(50) NOT NULL,
        is_verified BOOLEAN DEFAULT false,
        created_at TIMESTAMPTZ DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS ix_abi_paper_type ON agent_blackboard_items(paper_id, item_type);

    CREATE TABLE IF NOT EXISTS paper_extractions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
        extraction_type VARCHAR(30) NOT NULL,
        value_json JSONB NOT NULL,
        evidence_refs JSONB,
        producer_run_id UUID REFERENCES agent_runs(id),
        extraction_version SMALLINT DEFAULT 1,
        review_status VARCHAR(20) DEFAULT 'auto',
        created_at TIMESTAMPTZ DEFAULT now(),
        updated_at TIMESTAMPTZ DEFAULT now(),
        UNIQUE(paper_id, extraction_type, extraction_version)
    );

    CREATE TABLE IF NOT EXISTS reference_role_maps (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
        ref_index VARCHAR(10),
        ref_title TEXT,
        ref_arxiv_id VARCHAR(30),
        ref_candidate_id UUID REFERENCES paper_candidates(id),
        ref_paper_id UUID REFERENCES papers(id),
        role VARCHAR(30) NOT NULL,
        role_confidence FLOAT,
        where_mentioned TEXT[] DEFAULT '{}',
        mention_count SMALLINT DEFAULT 1,
        recommended_ingest_level VARCHAR(20),
        recommendation_reason TEXT,
        evidence_refs JSONB,
        created_at TIMESTAMPTZ DEFAULT now()
    );

    -- Metadata observations (sparse column replacement)
    CREATE TABLE IF NOT EXISTS metadata_observations (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        entity_type VARCHAR(50) NOT NULL,
        entity_id UUID NOT NULL,
        field_name VARCHAR(100) NOT NULL,
        value_json JSONB NOT NULL,
        source VARCHAR(50) NOT NULL,
        source_url TEXT,
        observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        confidence FLOAT DEFAULT 0.5,
        authority_rank SMALLINT DEFAULT 5,
        conflict_group_id UUID
    );
    CREATE INDEX IF NOT EXISTS idx_obs_entity ON metadata_observations(entity_type, entity_id);
    CREATE INDEX IF NOT EXISTS idx_obs_field ON metadata_observations(entity_type, entity_id, field_name);

    CREATE TABLE IF NOT EXISTS canonical_paper_metadata (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE UNIQUE,
        resolved_fields JSONB DEFAULT '{}',
        last_resolved_at TIMESTAMPTZ DEFAULT now()
    );

    -- Taxonomy DAG (Layer A)
    CREATE TABLE IF NOT EXISTS taxonomy_nodes (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        name VARCHAR(200) NOT NULL,
        name_zh VARCHAR(200),
        dimension VARCHAR(50) NOT NULL,
        aliases TEXT[],
        description TEXT,
        version SMALLINT DEFAULT 1,
        status VARCHAR(20) DEFAULT 'candidate',
        sort_order SMALLINT DEFAULT 0,
        created_at TIMESTAMPTZ DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS idx_taxnode_dimension ON taxonomy_nodes(dimension);
    CREATE INDEX IF NOT EXISTS idx_taxnode_name ON taxonomy_nodes(name);

    CREATE TABLE IF NOT EXISTS taxonomy_edges (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        parent_id UUID NOT NULL,
        child_id UUID NOT NULL,
        relation_type VARCHAR(30) NOT NULL,
        confidence FLOAT DEFAULT 1.0,
        evidence_ids TEXT[],
        UNIQUE(parent_id, child_id, relation_type)
    );
    CREATE INDEX IF NOT EXISTS idx_taxedge_parent ON taxonomy_edges(parent_id);
    CREATE INDEX IF NOT EXISTS idx_taxedge_child ON taxonomy_edges(child_id);

    CREATE TABLE IF NOT EXISTS paper_facets (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        paper_id UUID NOT NULL,
        node_id UUID NOT NULL,
        facet_role VARCHAR(50) NOT NULL,
        confidence FLOAT DEFAULT 1.0,
        source VARCHAR(30),
        evidence_refs JSONB,
        UNIQUE(paper_id, node_id, facet_role)
    );
    CREATE INDEX IF NOT EXISTS idx_facet_paper ON paper_facets(paper_id);
    CREATE INDEX IF NOT EXISTS idx_facet_node ON paper_facets(node_id);

    -- Problem nodes
    CREATE TABLE IF NOT EXISTS problem_nodes (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        title TEXT NOT NULL,
        description TEXT,
        domain_dimension VARCHAR(50),
        parent_problem_id UUID,
        status VARCHAR(20) DEFAULT 'active',
        created_at TIMESTAMPTZ DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS problem_claims (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
        problem_id UUID REFERENCES problem_nodes(id),
        claim_text TEXT NOT NULL,
        is_primary BOOLEAN DEFAULT true,
        confidence FLOAT,
        created_at TIMESTAMPTZ DEFAULT now()
    );

    -- Method DAG (Layer B)
    CREATE TABLE IF NOT EXISTS method_nodes (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        name VARCHAR(200) NOT NULL,
        name_zh VARCHAR(200),
        type VARCHAR(50) NOT NULL,
        domain VARCHAR(100),
        canonical_paper_id UUID,
        version VARCHAR(20),
        maturity VARCHAR(30) DEFAULT 'seed',
        description TEXT,
        promotion_criteria JSONB,
        parent_method_id UUID REFERENCES method_nodes(id),
        aliases TEXT[],
        downstream_count SMALLINT DEFAULT 0,
        created_at TIMESTAMPTZ DEFAULT now(),
        updated_at TIMESTAMPTZ DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS idx_method_name ON method_nodes(name);

    CREATE TABLE IF NOT EXISTS method_edges (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        parent_method_id UUID NOT NULL,
        child_method_id UUID NOT NULL,
        relation_type VARCHAR(30) NOT NULL,
        scope_facet_ids TEXT[],
        changed_slot_ids TEXT[],
        delta_description TEXT,
        evidence_refs JSONB,
        confidence FLOAT DEFAULT 0.5,
        status VARCHAR(20) DEFAULT 'candidate',
        created_at TIMESTAMPTZ DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS method_applications (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        method_id UUID NOT NULL REFERENCES method_nodes(id),
        paper_id UUID NOT NULL REFERENCES papers(id),
        role VARCHAR(30) NOT NULL,
        created_at TIMESTAMPTZ DEFAULT now()
    );

    -- Graph candidates (staging)
    CREATE TABLE IF NOT EXISTS graph_node_candidates (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        paper_id UUID REFERENCES papers(id),
        candidate_id UUID REFERENCES paper_candidates(id),
        node_type VARCHAR(20) NOT NULL,
        name VARCHAR(200) NOT NULL,
        name_zh VARCHAR(200),
        one_liner TEXT,
        promotion_score FLOAT,
        promotion_breakdown JSONB,
        status VARCHAR(20) DEFAULT 'candidate',
        promoted_entity_type VARCHAR(30),
        promoted_entity_id UUID,
        evidence_refs JSONB,
        confidence FLOAT,
        created_at TIMESTAMPTZ DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS graph_edge_candidates (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        paper_id UUID REFERENCES papers(id),
        source_entity_type VARCHAR(30) NOT NULL,
        source_entity_id UUID NOT NULL,
        target_entity_type VARCHAR(30) NOT NULL,
        target_entity_id UUID NOT NULL,
        relation_type VARCHAR(50) NOT NULL,
        slot_name VARCHAR(100),
        confidence_score FLOAT,
        confidence_breakdown JSONB,
        one_liner TEXT,
        status VARCHAR(20) DEFAULT 'candidate',
        promoted_edge_id UUID,
        evidence_refs JSONB,
        created_at TIMESTAMPTZ DEFAULT now()
    );

    -- KB profiles
    CREATE TABLE IF NOT EXISTS kb_node_profiles (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        entity_type VARCHAR(30) NOT NULL,
        entity_id UUID NOT NULL,
        profile_kind VARCHAR(20) DEFAULT 'page',
        lang VARCHAR(5) DEFAULT 'zh',
        one_liner TEXT,
        short_intro_md TEXT,
        detailed_md TEXT,
        structured_json JSONB,
        evidence_refs JSONB,
        generated_by_run_id UUID REFERENCES agent_runs(id),
        model_name VARCHAR(50),
        prompt_version VARCHAR(20),
        profile_version SMALLINT DEFAULT 1,
        review_status VARCHAR(20) DEFAULT 'auto',
        staleness_trigger_count SMALLINT DEFAULT 0,
        created_at TIMESTAMPTZ DEFAULT now(),
        updated_at TIMESTAMPTZ DEFAULT now(),
        UNIQUE(entity_type, entity_id, profile_kind, lang)
    );

    CREATE TABLE IF NOT EXISTS kb_edge_profiles (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        source_entity_type VARCHAR(30) NOT NULL,
        source_entity_id UUID NOT NULL,
        target_entity_type VARCHAR(30) NOT NULL,
        target_entity_id UUID NOT NULL,
        relation_type VARCHAR(50) NOT NULL,
        edge_table VARCHAR(30),
        edge_id UUID,
        lang VARCHAR(5) DEFAULT 'zh',
        one_liner TEXT,
        relation_summary TEXT,
        source_context TEXT,
        target_context TEXT,
        display_priority SMALLINT DEFAULT 5,
        evidence_refs JSONB,
        generated_by_run_id UUID REFERENCES agent_runs(id),
        review_status VARCHAR(20) DEFAULT 'auto',
        created_at TIMESTAMPTZ DEFAULT now(),
        updated_at TIMESTAMPTZ DEFAULT now()
    );

    -- Paper reports
    CREATE TABLE IF NOT EXISTS paper_reports (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
        report_version SMALLINT DEFAULT 1,
        title_zh TEXT,
        title_en TEXT,
        generated_by_run_id UUID REFERENCES agent_runs(id),
        model_name VARCHAR(50),
        prompt_version VARCHAR(20),
        review_status VARCHAR(20) DEFAULT 'auto',
        created_at TIMESTAMPTZ DEFAULT now(),
        updated_at TIMESTAMPTZ DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS paper_report_sections (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        report_id UUID NOT NULL REFERENCES paper_reports(id) ON DELETE CASCADE,
        section_type VARCHAR(30) NOT NULL,
        title TEXT,
        body_md TEXT,
        evidence_refs JSONB,
        order_index SMALLINT NOT NULL,
        created_at TIMESTAMPTZ DEFAULT now()
    );

    -- Review queue
    CREATE TABLE IF NOT EXISTS review_queue_items (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        item_type VARCHAR(30) NOT NULL,
        entity_type VARCHAR(30) NOT NULL,
        entity_id UUID NOT NULL,
        priority_score FLOAT DEFAULT 50,
        status VARCHAR(20) DEFAULT 'pending',
        assigned_to VARCHAR(100),
        resolution JSONB,
        created_at TIMESTAMPTZ DEFAULT now(),
        resolved_at TIMESTAMPTZ
    );

    -- Evidence items (fine-grained)
    CREATE TABLE IF NOT EXISTS evidence_items (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
        item_type VARCHAR(30) NOT NULL,
        page_num SMALLINT,
        bbox JSONB,
        text_content TEXT,
        formula_latex TEXT,
        figure_object_key TEXT,
        confidence FLOAT,
        created_at TIMESTAMPTZ DEFAULT now()
    );

    -- DeltaCard lineage
    CREATE TABLE IF NOT EXISTS delta_card_lineage (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        parent_id UUID NOT NULL REFERENCES delta_cards(id),
        child_id UUID NOT NULL REFERENCES delta_cards(id),
        relation_type VARCHAR(30) NOT NULL,
        changed_slots TEXT[],
        evidence TEXT,
        created_at TIMESTAMPTZ DEFAULT now(),
        UNIQUE(parent_id, child_id, relation_type)
    );

    -- Paper bottleneck claims
    CREATE TABLE IF NOT EXISTS paper_bottleneck_claims (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
        bottleneck_id UUID REFERENCES project_bottlenecks(id),
        raw_title TEXT,
        claim_text TEXT NOT NULL,
        is_primary BOOLEAN DEFAULT true,
        is_fundamental BOOLEAN,
        confidence FLOAT,
        source VARCHAR(30) DEFAULT 'system_inferred',
        created_at TIMESTAMPTZ DEFAULT now()
    );

    -- Venue papers (crawl cache)
    CREATE TABLE IF NOT EXISTS venue_papers (
        id SERIAL PRIMARY KEY,
        arxiv_id VARCHAR(30),
        doi VARCHAR(200),
        openreview_forum_id VARCHAR(200),
        title TEXT,
        title_normalized TEXT,
        authors JSONB,
        abstract TEXT,
        venue VARCHAR(100),
        year SMALLINT,
        conf_year VARCHAR(30),
        decision VARCHAR(50),
        acceptance_type VARCHAR(30),
        paper_link TEXT,
        pdf_url TEXT,
        code_url TEXT,
        project_url TEXT,
        keywords TEXT[],
        topic VARCHAR(200),
        extra_data JSONB,
        pdf_object_key VARCHAR(500),
        pdf_size_bytes INT,
        pdf_checksum VARCHAR(64),
        pdf_downloaded_at TIMESTAMPTZ,
        crawled_at TIMESTAMPTZ DEFAULT now(),
        updated_at TIMESTAMPTZ DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS ix_vp_arxiv ON venue_papers(arxiv_id);
    CREATE INDEX IF NOT EXISTS ix_vp_doi ON venue_papers(doi);
    CREATE INDEX IF NOT EXISTS ix_vp_title ON venue_papers(title_normalized);
    CREATE INDEX IF NOT EXISTS ix_vp_conf ON venue_papers(conf_year);

    -- Taxonomy versions (audit)
    CREATE TABLE IF NOT EXISTS taxonomy_versions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        entity_type VARCHAR(30) NOT NULL,
        entity_id UUID NOT NULL,
        action VARCHAR(20) NOT NULL,
        before_snapshot JSONB,
        after_snapshot JSONB,
        created_at TIMESTAMPTZ DEFAULT now()
    );

    -- Canonical ideas
    CREATE TABLE IF NOT EXISTS canonical_ideas (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        title TEXT NOT NULL,
        description TEXT,
        method_node_id UUID,
        status VARCHAR(20) DEFAULT 'draft',
        created_at TIMESTAMPTZ DEFAULT now(),
        updated_at TIMESTAMPTZ DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS contribution_to_canonical_idea (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        canonical_idea_id UUID REFERENCES canonical_ideas(id) ON DELETE CASCADE,
        paper_id UUID REFERENCES papers(id) ON DELETE CASCADE,
        delta_card_id UUID REFERENCES delta_cards(id),
        contribution_type VARCHAR(30),
        evidence_refs JSONB,
        created_at TIMESTAMPTZ DEFAULT now()
    );
    """)
    print("  Created 30+ V6 tables")

    # ── 5. Verify ─────────────────────────────────────────────────
    print("\n[5/5] Verifying...")
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name;
    """)
    tables = [r[0] for r in cur.fetchall()]

    required = [
        'papers', 'paper_analyses', 'delta_cards', 'evidence_units',
        'graph_nodes', 'graph_assertions',
        'agent_runs', 'agent_blackboard_items',
        'paper_candidates', 'candidate_scores',
        'taxonomy_nodes', 'taxonomy_edges', 'paper_facets',
        'method_nodes', 'method_edges',
        'kb_node_profiles', 'kb_edge_profiles',
        'paper_reports', 'paper_report_sections',
        'metadata_observations', 'venue_papers',
        'graph_node_candidates', 'graph_edge_candidates',
    ]

    missing = [t for t in required if t not in tables]
    if missing:
        print(f"  ERROR: Missing tables: {missing}")
        sys.exit(1)

    # Check papers columns
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'papers' ORDER BY ordinal_position;
    """)
    cols = [r[0] for r in cur.fetchall()]
    needed = ['acceptance_type', 'ring', 'current_delta_card_id', 'method_family', 'cited_by_count']
    missing_cols = [c for c in needed if c not in cols]
    if missing_cols:
        print(f"  ERROR: Missing papers columns: {missing_cols}")
        sys.exit(1)

    print(f"  OK: {len(tables)} tables, all required columns present")
    print("\n=== Schema patch complete ===")

    cur.close()
    conn.close()


if __name__ == "__main__":
    run_patch()
