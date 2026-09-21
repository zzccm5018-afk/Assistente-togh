-- setup_tables.sql
-- =================
-- Rode isto UMA VEZ no seu projeto Supabase: painel > SQL Editor > New query
-- > cole tudo > Run.
--
-- Cria as 4 tabelas que o sistema inteiro precisa:
--   1. conversations   -> já esperada pelo main.py (log_conversation)
--   2. feedback        -> já esperada pelo main.py (rota /feedback)
--   3. model_config     -> nova: guarda qual modelo está ativo (hot-swap)
--   4. training_runs    -> nova, opcional: histórico dos treinos automáticos

-- ---------------------------------------------------------------------------
-- 1. conversations
-- ---------------------------------------------------------------------------
create table if not exists conversations (
    id uuid primary key,
    user_id text not null,
    channel text not null,
    user_message text not null,
    answer text not null,
    sources jsonb,
    created_at timestamptz default now()
);

create index if not exists idx_conversations_created_at
    on conversations (created_at);

-- ---------------------------------------------------------------------------
-- 2. feedback
-- ---------------------------------------------------------------------------
create table if not exists feedback (
    id uuid primary key default gen_random_uuid(),
    conversation_id uuid references conversations (id) on delete cascade,
    rating int not null check (rating in (-1, 0, 1)),
    comment text,
    created_at timestamptz default now()
);

create index if not exists idx_feedback_conversation_id
    on feedback (conversation_id);

-- ---------------------------------------------------------------------------
-- 3. model_config  (hot-swap do modelo ativo, lido pelo main.py)
-- ---------------------------------------------------------------------------
create table if not exists model_config (
    id int primary key default 1,
    active_model text not null,
    updated_at timestamptz default now(),
    constraint singleton check (id = 1)
);

-- Insere o modelo base como valor inicial (só roda se a tabela estiver vazia)
insert into model_config (id, active_model)
values (1, 'meta-llama/Llama-3.3-70B-Instruct-Turbo')
on conflict (id) do nothing;

-- ---------------------------------------------------------------------------
-- 4. training_runs  (histórico do pipeline de aprendizado contínuo)
-- ---------------------------------------------------------------------------
create table if not exists training_runs (
    id uuid primary key default gen_random_uuid(),
    started_at timestamptz default now(),
    finished_at timestamptz,
    status text,                 -- 'success' | 'failed' | 'skipped'
    examples_count int,
    together_job_id text,
    model_id text,
    notes text
);

create index if not exists idx_training_runs_status_finished_at
    on training_runs (status, finished_at desc);
