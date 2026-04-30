-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ─────────────────────────────────────────────────────────────
-- Document chunks: raw ingested content + embeddings
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT        NOT NULL,          -- google_drive, talon_docs, local
    source_id   TEXT,                          -- Drive file ID or doc URL slug
    title       TEXT,
    content     TEXT        NOT NULL,
    metadata    JSONB       DEFAULT '{}',
    embedding   vector(768),                   -- Gemini gemini-embedding-001 (output_dimensionality=768)
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for cosine similarity search (pgvector 0.7+ — faster than IVFFlat,
-- no training step required, better recall at high query rates)
CREATE INDEX IF NOT EXISTS idx_documents_embedding
    ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
-- m=16: graph connectivity (higher = better recall, more RAM)
-- ef_construction=64: build-time search width (higher = better recall, slower build)

CREATE INDEX IF NOT EXISTS idx_documents_source
    ON documents (source);

-- ─────────────────────────────────────────────────────────────
-- Generated presentations: track outputs per prospect
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS presentations (
    id              BIGSERIAL PRIMARY KEY,
    company_name    TEXT        NOT NULL,
    industry        TEXT,
    use_case        TEXT,
    html_content    TEXT,
    github_pr_url   TEXT,
    vercel_url      TEXT,
    status          TEXT        DEFAULT 'draft',  -- draft | deployed | archived
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────
-- User Insights (sales_playbook) for long-term coaching
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_insights (
    id          BIGSERIAL PRIMARY KEY,
    user_id     TEXT        NOT NULL,
    namespace   TEXT        NOT NULL,
    thread_id   TEXT,
    content     TEXT        NOT NULL,
    tags        JSONB       DEFAULT '[]',
    embedding   vector(768),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_insights_user_namespace
    ON user_insights (user_id, namespace);

CREATE INDEX IF NOT EXISTS idx_user_insights_embedding
    ON user_insights USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ─────────────────────────────────────────────────────────────
-- Agent run logs: trace each LangGraph invocation
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS agent_runs (
    id              BIGSERIAL PRIMARY KEY,
    presentation_id BIGINT      REFERENCES presentations(id),
    agent_name      TEXT        NOT NULL,
    input_summary   TEXT,
    output_summary  TEXT,
    tokens_used     INT,
    duration_ms     INT,
    error           TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
