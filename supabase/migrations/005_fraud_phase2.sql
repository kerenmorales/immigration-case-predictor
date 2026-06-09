-- ============================================================
-- Fraud Check — Phase 2 additions
-- Adds extracted_entities + domain_age_checks JSONB columns
-- so we can show users actionable verification info.
-- ============================================================

alter table fraud_checks
  add column if not exists extracted_entities jsonb default '{}'::jsonb,
  add column if not exists domain_age_checks jsonb default '[]'::jsonb;
