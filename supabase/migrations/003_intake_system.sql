-- ============================================================
-- AI Intake System ($29 — AI conversational intake + 30-min consult)
-- ============================================================

-- 1. INTAKE SESSIONS — one per paid intake
create table if not exists intake_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade,

  -- Application context
  application_type text not null,  -- 'spousal_sponsorship', 'work_permit', 'open_ended'
  language text default 'es',       -- 'es' or 'en'

  -- Payment tracking
  is_paid boolean default false,
  stripe_checkout_session_id text,
  stripe_payment_intent_id text,
  paid_at timestamptz,

  -- Conversation state
  status text default 'pending_payment',
    -- 'pending_payment' | 'in_progress' | 'awaiting_documents' | 'completed' | 'abandoned'
  conversation_history jsonb default '[]'::jsonb,
    -- Array of {role: 'user'|'assistant', content: '...', timestamp: ...}
  collected_facts jsonb default '{}'::jsonb,
    -- Structured facts the AI has extracted from the conversation
  red_flags jsonb default '[]'::jsonb,
  strengths jsonb default '[]'::jsonb,

  -- Final output
  user_summary_es text,
  user_summary_en text,
  lawyer_summary text,
  next_steps jsonb default '[]'::jsonb,

  -- Booking
  calendly_link_used text,
  consult_booked_at timestamptz,

  -- Timestamps
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  completed_at timestamptz
);

create index if not exists idx_intake_user on intake_sessions(user_id);
create index if not exists idx_intake_status on intake_sessions(status);
create index if not exists idx_intake_paid on intake_sessions(is_paid) where is_paid = true;

-- 2. INTAKE DOCUMENTS — files uploaded for the case
create table if not exists intake_documents (
  id uuid primary key default gen_random_uuid(),
  intake_session_id uuid references intake_sessions(id) on delete cascade,
  user_id uuid references auth.users(id) on delete cascade,

  filename text not null,
  storage_path text not null,        -- path in Supabase Storage
  mime_type text,
  size_bytes bigint,
  document_category text,             -- 'passport', 'marriage_certificate', 'other', etc.

  -- Optional AI analysis of the document
  ai_extracted_text text,
  ai_observations text,

  uploaded_at timestamptz default now()
);

create index if not exists idx_intake_docs_session on intake_documents(intake_session_id);

-- 3. RLS — users only see their own intakes
alter table intake_sessions enable row level security;
alter table intake_documents enable row level security;

drop policy if exists "Users see own intakes" on intake_sessions;
create policy "Users see own intakes" on intake_sessions
  for all using (auth.uid() = user_id);

drop policy if exists "Users see own intake docs" on intake_documents;
create policy "Users see own intake docs" on intake_documents
  for all using (auth.uid() = user_id);

-- 4. Helper: count files per intake (for the 10-file cap)
create or replace function intake_file_count(session_id uuid) returns int
language sql stable as $$
  select count(*)::int from intake_documents where intake_session_id = session_id;
$$;

-- 5. Helper: total bytes per intake (for the 50MB cap)
create or replace function intake_total_bytes(session_id uuid) returns bigint
language sql stable as $$
  select coalesce(sum(size_bytes), 0) from intake_documents where intake_session_id = session_id;
$$;
