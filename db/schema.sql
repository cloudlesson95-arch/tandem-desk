-- Tandem Desk schema. Idempotent: safe to apply on every setup run.
CREATE EXTENSION IF NOT EXISTS vector;

-- Policy passages for retrieval. id is the content hash, so re-ingesting is a no-op.
CREATE TABLE IF NOT EXISTS chunks (
    id         TEXT PRIMARY KEY,
    doc        TEXT NOT NULL,
    title      TEXT NOT NULL,
    content    TEXT NOT NULL,
    embedding  vector(768) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS customers (
    id    TEXT PRIMARY KEY,
    name  TEXT NOT NULL,
    email TEXT NOT NULL,
    phone TEXT NOT NULL,
    tier  TEXT NOT NULL CHECK (tier IN ('standard', 'plus'))
);

CREATE TABLE IF NOT EXISTS rentals (
    id          TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    bike        TEXT NOT NULL,
    start_date  DATE NOT NULL,
    end_date    DATE NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('booked', 'active', 'returned', 'cancelled')),
    daily_rate  NUMERIC(6, 2) NOT NULL,
    extended_days INT NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS rentals_customer_idx ON rentals (customer_id);

-- Long-term memory: durable preferences and episode summaries, stored already redacted.
CREATE TABLE IF NOT EXISTS memories (
    id          BIGSERIAL PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL CHECK (kind IN ('preference', 'episode')),
    content     TEXT NOT NULL,
    embedding   vector(768) NOT NULL,
    thread_id   TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS memories_customer_idx ON memories (customer_id, created_at DESC);

-- Every proposed write action and the human decision on it.
CREATE TABLE IF NOT EXISTS action_log (
    id          BIGSERIAL PRIMARY KEY,
    customer_id TEXT NOT NULL,
    thread_id   TEXT NOT NULL,
    action      TEXT NOT NULL,
    args        JSONB NOT NULL,
    decision    TEXT NOT NULL CHECK (decision IN ('pending', 'approved', 'rejected', 'executed', 'failed')),
    result      JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS action_log_customer_idx ON action_log (customer_id, created_at DESC);

-- Demo guard counters: one row per (key, window start).
CREATE TABLE IF NOT EXISTS rate_limits (
    key          TEXT NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    count        INT NOT NULL DEFAULT 0,
    PRIMARY KEY (key, window_start)
);
