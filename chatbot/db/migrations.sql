-- ============================================================
-- Supabase DB Migrations
-- Run once in the Supabase SQL Editor:
--   https://supabase.com/dashboard/project/<your-project>/sql
-- ============================================================

-- Enable pgvector extension
create extension if not exists vector;

-- NOTE: If upgrading an existing DB, run this first:
-- alter table clients add column if not exists website_url text default '';
-- alter table clients add column if not exists is_active_demo boolean default false;
-- alter table clients add column if not exists workflow_instructions text default '';

-- ── clients table ────────────────────────────────────────────
-- Stores client config (replaces clients/*/config.yaml)
create table if not exists clients (
    client_id         text primary key,
    business_name     text not null,
    hardware_tier     text not null default 'A',
    tone              text not null default 'friendly',
    website_url       text not null default '',
    is_active_demo    boolean not null default false,
    refusal_message   text not null default 'I can only answer questions about {business_name}.',
    retrieval_top_k   int  not null default 5,
    score_threshold   float not null default 0.35,
    chunk_size        int  not null default 512,
    chunk_overlap     int  not null default 64,
    max_history_turns int  not null default 6,
    workflow_instructions text not null default '',
    created_at        timestamptz default now(),
    updated_at        timestamptz default now()
);

-- ── sessions table ───────────────────────────────────────────
-- Stores chat session history
create table if not exists sessions (
    session_id text primary key,
    client_id  text not null references clients(client_id) on delete cascade,
    messages   jsonb not null default '[]'::jsonb,
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

-- ── chunks table ─────────────────────────────────────────────
-- Stores document chunks + embeddings for all clients
-- BGE-small dense vector dimension = 384
create table if not exists chunks (
    id         bigserial primary key,
    client_id  text not null references clients(client_id) on delete cascade,
    chunk_id   text not null unique,
    text       text not null,
    source     text not null default '',
    embedding  vector(384) not null,
    created_at timestamptz default now()
);

-- Index for fast similarity search
create index if not exists chunks_embedding_idx
    on chunks using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

-- Index for client partitioning
create index if not exists chunks_client_id_idx on chunks(client_id);

-- ── match_chunks function ─────────────────────────────────────
-- Called by Supabase RPC for pgvector similarity search
create or replace function match_chunks(
    p_client_id     text,
    query_embedding vector(384),
    match_count     int     default 5,
    match_threshold float   default 0.35
)
returns table (
    chunk_id  text,
    text      text,
    source    text,
    score     float
)
language sql stable
as $$
    select
        c.chunk_id,
        c.text,
        c.source,
        1 - (c.embedding <=> query_embedding) as score
    from chunks c
    where
        c.client_id = p_client_id
        and 1 - (c.embedding <=> query_embedding) >= match_threshold
    order by c.embedding <=> query_embedding
    limit match_count;
$$;

-- ── RLS policies (optional — enable for production) ───────────
-- alter table clients enable row level security;
-- alter table chunks  enable row level security;
-- For now keep RLS off — protect via SUPABASE_KEY (service role key in backend)

-- ── Updated_at trigger ────────────────────────────────────────
create or replace function set_updated_at()
returns trigger language plpgsql as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists clients_updated_at on clients;
create trigger clients_updated_at
    before update on clients
    for each row execute procedure set_updated_at();
