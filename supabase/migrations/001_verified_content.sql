-- ============================================================
-- Verified Content System
-- ============================================================
-- This system ensures all IRCC info shown to paid users is
-- explicitly verified by a real person (you) and auto-flagged
-- when it gets stale.
--
-- Run this in your Supabase SQL Editor.
-- ============================================================

-- 1. CONTENT CATEGORIES — what types of info we track
create table if not exists content_categories (
  id text primary key,
  name_en text not null,
  name_es text not null,
  description text
);

insert into content_categories (id, name_en, name_es, description) values
  ('fee', 'Government Fee', 'Tarifa Gubernamental', 'IRCC application fees'),
  ('processing_time', 'Processing Time', 'Tiempo de Procesamiento', 'IRCC processing time estimates'),
  ('form_field', 'Form Field', 'Campo de Formulario', 'IRCC form field definitions and instructions'),
  ('eligibility', 'Eligibility Rule', 'Regla de Elegibilidad', 'Eligibility requirements for applications'),
  ('document', 'Required Document', 'Documento Requerido', 'Documents required for applications'),
  ('faq', 'FAQ Entry', 'Pregunta Frecuente', 'Frequently asked questions and answers'),
  ('warning', 'Warning / Red Flag', 'Advertencia', 'Common pitfalls and red flags')
on conflict (id) do nothing;

-- 2. APPLICATION TYPES — what IRCC applications we cover
create table if not exists application_types (
  id text primary key,
  ircc_code text,
  name_en text not null,
  name_es text not null,
  description_en text,
  description_es text,
  is_active boolean default true,
  is_paid_only boolean default false,
  display_order int default 0
);

insert into application_types (id, ircc_code, name_en, name_es, description_en, description_es, is_paid_only, display_order) values
  ('work_permit', 'IMM 1295', 'Work Permit (Outside Canada)', 'Permiso de Trabajo (Fuera de Canadá)', 'Apply for a Canadian work permit from outside Canada', 'Solicitar un permiso de trabajo canadiense desde fuera de Canadá', true, 1),
  ('work_permit_extension', 'IMM 5710', 'Work Permit Extension', 'Extensión de Permiso de Trabajo', 'Extend your existing Canadian work permit', 'Extender su permiso de trabajo canadiense existente', true, 2),
  ('spousal_sponsorship', 'IMM 1344', 'Spousal Sponsorship', 'Patrocinio Conyugal', 'Sponsor your spouse or partner for permanent residence', 'Patrocinar a su esposo/a o pareja para residencia permanente', true, 3),
  ('visitor_visa', 'IMM 5257', 'Visitor Visa', 'Visa de Visitante', 'Apply to visit Canada', 'Solicitar visitar Canadá', true, 4),
  ('study_permit', 'IMM 1294', 'Study Permit', 'Permiso de Estudio', 'Apply to study in Canada', 'Solicitar estudiar en Canadá', true, 5),
  ('pr_card_renewal', 'IMM 5444', 'PR Card Renewal', 'Renovación de Tarjeta PR', 'Renew your Permanent Resident card', 'Renovar su tarjeta de Residente Permanente', true, 6),
  ('citizenship', 'CIT 0002', 'Citizenship Application', 'Solicitud de Ciudadanía', 'Apply for Canadian citizenship', 'Solicitar ciudadanía canadiense', true, 7)
on conflict (id) do nothing;

-- 3. THE VERIFIED CONTENT TABLE
create table if not exists verified_content (
  id uuid primary key default gen_random_uuid(),
  application_type_id text references application_types(id),
  category_id text references content_categories(id) not null,
  key text not null,
  content_en text not null,
  content_es text not null,
  metadata jsonb default '{}'::jsonb,
  source_url text,
  source_quote text,
  verified_by text,
  verified_at timestamptz default now(),
  expires_at timestamptz default (now() + interval '90 days'),
  is_published boolean default true,
  needs_review boolean default false,
  review_notes text,
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  unique(application_type_id, category_id, key)
);

create index if not exists idx_verified_content_app on verified_content(application_type_id);
create index if not exists idx_verified_content_published on verified_content(is_published) where is_published = true;
create index if not exists idx_verified_content_needs_review on verified_content(needs_review) where needs_review = true;
create index if not exists idx_verified_content_expires on verified_content(expires_at);

-- 4. SOURCE MONITORING
create table if not exists monitored_sources (
  id uuid primary key default gen_random_uuid(),
  url text unique not null,
  description text,
  last_checked_at timestamptz,
  last_content_hash text,
  last_change_detected_at timestamptz,
  is_active boolean default true,
  notify_email text,
  created_at timestamptz default now()
);

-- 5. CHANGE LOG
create table if not exists verified_content_history (
  id uuid primary key default gen_random_uuid(),
  content_id uuid references verified_content(id) on delete cascade,
  change_type text,
  previous_content jsonb,
  new_content jsonb,
  changed_by text,
  changed_at timestamptz default now()
);

-- ============================================================
-- USER DATA SAFETY — make existing tables bulletproof
-- ============================================================

do $$
declare
  t text;
begin
  for t in select unnest(array[
    'visa_forms', 'predictions', 'proof_entries', 'photo_album_data',
    'form_tracker', 'sponsorship_forms', 'client_intakes'
  ])
  loop
    execute format('alter table %I add column if not exists deleted_at timestamptz', t);
    execute format('alter table %I add column if not exists version int default 1', t);
    execute format('create index if not exists idx_%I_user_active on %I(user_id) where deleted_at is null', t, t);
  end loop;
end $$;

-- ============================================================
-- USER PROFILES — for paid subscriptions and Stripe linkage
-- ============================================================
create table if not exists user_profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text,
  full_name text,
  preferred_language text default 'es',
  phone text,
  subscription_status text default 'free',
  subscription_tier text,
  stripe_customer_id text,
  stripe_subscription_id text,
  trial_ends_at timestamptz,
  subscription_current_period_end timestamptz,
  preferred_apps text[] default array[]::text[],
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  last_login_at timestamptz
);

create index if not exists idx_user_profiles_stripe_customer on user_profiles(stripe_customer_id);
create index if not exists idx_user_profiles_subscription_status on user_profiles(subscription_status);

-- Auto-create profile on user signup
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
  insert into public.user_profiles (id, email)
  values (new.id, new.email)
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- ============================================================
-- ROW LEVEL SECURITY
-- ============================================================

alter table user_profiles enable row level security;

drop policy if exists "Users can view own profile" on user_profiles;
create policy "Users can view own profile" on user_profiles
  for select using (auth.uid() = id);

drop policy if exists "Users can update own profile" on user_profiles;
create policy "Users can update own profile" on user_profiles
  for update using (auth.uid() = id);

alter table verified_content enable row level security;
alter table application_types enable row level security;
alter table content_categories enable row level security;

drop policy if exists "Verified content readable by all" on verified_content;
create policy "Verified content readable by all" on verified_content
  for select using (is_published = true);

drop policy if exists "Application types readable by all" on application_types;
create policy "Application types readable by all" on application_types
  for select using (true);

drop policy if exists "Content categories readable by all" on content_categories;
create policy "Content categories readable by all" on content_categories
  for select using (true);

do $$
declare
  t text;
begin
  for t in select unnest(array[
    'visa_forms', 'predictions', 'proof_entries', 'photo_album_data',
    'form_tracker', 'sponsorship_forms', 'client_intakes'
  ])
  loop
    execute format('alter table %I enable row level security', t);
    execute format('drop policy if exists "Users access own %I" on %I', t, t);
    execute format('create policy "Users access own %I" on %I for all using (auth.uid() = user_id)', t, t);
  end loop;
end $$;

-- ============================================================
-- HELPER VIEW for the admin dashboard
-- ============================================================

create or replace view content_health as
select
  at.name_en as application,
  at.is_active,
  count(vc.id) filter (where vc.is_published) as published_items,
  count(vc.id) filter (where vc.needs_review) as needs_review,
  count(vc.id) filter (where vc.expires_at < now()) as expired_items,
  min(vc.expires_at) as next_expiration
from application_types at
left join verified_content vc on vc.application_type_id = at.id
group by at.id, at.name_en, at.is_active
order by at.display_order;
