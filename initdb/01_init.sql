-- ============================================================================
-- 01_init.sql - Baseline Database Initialization for Arabic IR Research
-- Database: arabic_ir (automatically created and selected by POSTGRES_DB)
-- Milestone: M1 (Environment & Baseline Configuration)
-- ============================================================================

-- Ensure UTF8 client encoding during script execution
SET client_encoding = 'UTF8';

-- 1. Enable pg_trgm (Trigram matching for System S5)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 2. Enable unaccent (Text search dictionary for accent/diacritic removal)
CREATE EXTENSION IF NOT EXISTS unaccent;

-- Verification commentary logged to container stdout
DO $$
BEGIN
    RAISE NOTICE 'Arabic IR Baseline Initialization Completed: Extensions pg_trgm and unaccent enabled.';
END $$;
