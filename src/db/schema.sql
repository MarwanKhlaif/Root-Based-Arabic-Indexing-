-- ============================================================================
-- schema.sql - Core Database Schema for Arabic IR Research (PostgreSQL 16)
-- Target: database `arabic_ir`
-- Supports Systems S0 through S5 Full-Text Indexing Configurations
-- ============================================================================

SET client_encoding = 'UTF8';

-- 1. Enable Required Extensions
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE EXTENSION IF NOT EXISTS pg_prewarm;

-- 2. Core Corpus Documents Table
-- Holds 100,000 Wikipedia articles and accommodates all experimental tsvector representations
CREATE TABLE IF NOT EXISTS docs (
    id BIGSERIAL PRIMARY KEY,
    doc_id TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    normalized_body TEXT NOT NULL,
    s0_tsvector tsvector,
    s1_tsvector tsvector,
    s2_tsvector tsvector,
    s3_tsvector tsvector,
    s4_tsvector tsvector,
    s5_trgm_text TEXT,
    url TEXT,
    word_count INTEGER,
    char_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_docs_doc_id UNIQUE (doc_id)
);

-- Comments documenting schema contracts
COMMENT ON TABLE docs IS 'Arabic Wikipedia research corpus (100,000 clean articles, seed 42) for IR benchmarking.';
COMMENT ON COLUMN docs.doc_id IS 'Unique external Wikipedia article identifier.';
COMMENT ON COLUMN docs.normalized_body IS 'Text preprocessed via normalize.py (tashkeel/tatweel stripped, alif/ya/ta-marbuta unified).';
COMMENT ON COLUMN docs.s0_tsvector IS 'System S0: Surface simple tsvector (to_tsvector(''simple'', normalized_body)).';
COMMENT ON COLUMN docs.s1_tsvector IS 'System S1: Arabic simple tsvector with stopword filtering (arabic_simple).';
COMMENT ON COLUMN docs.s2_tsvector IS 'System S2: Larkey light10 affix-stripped stem tsvector.';
COMMENT ON COLUMN docs.s3_tsvector IS 'System S3: ISRI morphological pattern root tsvector.';
COMMENT ON COLUMN docs.s4_tsvector IS 'System S4: Khoja dictionary-validated root tsvector.';
COMMENT ON COLUMN docs.s5_trgm_text IS 'System S5: Normalized text indexed with pg_trgm character trigrams.';

-- ============================================================================
-- Deferred Index Creation Commands (Execute AFTER Bulk Ingestion)
-- ============================================================================
-- S0 Surface Baseline GIN Index:
-- CREATE INDEX IF NOT EXISTS idx_docs_s0_gin ON docs USING GIN (s0_tsvector);

-- S5 Trigram GIN Index:
-- CREATE INDEX IF NOT EXISTS idx_docs_s5_trgm_gin ON docs USING GIN (s5_trgm_text gin_trgm_ops);

-- Planner Statistics Refresh:
-- VACUUM ANALYZE docs;
