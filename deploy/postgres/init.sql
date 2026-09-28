-- ============================================================================
-- 智能客服平台 · PostgreSQL + pgvector 初始化
--
-- 🔴 doc_chunk 按 tenant_id LIST 分区，不是"一张大表 + metadata 过滤"：
--    共享一个近似索引时租户之间会互相影响召回（HNSW 图是全局的，
--    A 租户的向量会占掉 B 租户的搜索预算）。分区后每租户独立索引，
--    删租户也只是 DROP 分区。
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- 文档
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document (
    tenant_id     text        NOT NULL,
    doc_id        text        NOT NULL,
    kb_id         text        NOT NULL DEFAULT 'default',
    title         text        NOT NULL DEFAULT '',
    -- 内容指纹：重复入库时据此跳过重切分与重嵌入（嵌入是最贵的一步）
    content_hash  text        NOT NULL,
    chunk_count   int         NOT NULL DEFAULT 0,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, doc_id)
) PARTITION BY LIST (tenant_id);

-- ---------------------------------------------------------------------------
-- 切块。向量与词法索引都挂这里。
-- 🔴 UNIQUE(tenant_id, doc_id, chunk_no) 保证摄取幂等：重跑不产生重复块。
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS doc_chunk (
    id            uuid        NOT NULL,
    tenant_id     text        NOT NULL,
    doc_id        text        NOT NULL,
    kb_id         text        NOT NULL DEFAULT 'default',
    chunk_no      int         NOT NULL,
    content       text        NOT NULL,
    section_path  text,                    -- 标题路径：引用定位 + 上下文增强
    parent_id     uuid,                    -- 父子块
    -- jieba 分词后的空格分隔串。🔴 中文不分词直接 to_tsvector 会全 0 分。
    lexical_tsv   tsvector,
    metadata      jsonb       NOT NULL DEFAULT '{}'::jsonb,
    -- 维度必须与 model/registry.py 的 embed-main.dimensions 一致（bge-m3 = 1024）。
    -- 换嵌入模型 = 语义空间变了，必须重建全部向量，不能只改这个数字。
    embedding     vector(1024),
    created_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id),
    UNIQUE (tenant_id, doc_id, chunk_no)
) PARTITION BY LIST (tenant_id);

-- ---------------------------------------------------------------------------
-- 会话索引。checkpointer 存图状态，这里存"给人看的"元数据。
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS conversation (
    tenant_id       text        NOT NULL,
    conversation_id text        NOT NULL,
    user_id         text,
    title           text        NOT NULL DEFAULT '',
    agent_id        text        NOT NULL DEFAULT '',
    message_count   int         NOT NULL DEFAULT 0,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, conversation_id)
);
CREATE INDEX IF NOT EXISTS ix_conversation_recent
    ON conversation (tenant_id, updated_at DESC);

-- ---------------------------------------------------------------------------
-- 待确认操作单。🔴 写操作只产出操作单，没有绕过路径。
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pending_action (
    id          uuid        PRIMARY KEY,
    tenant_id   text        NOT NULL,
    user_id     text,
    tool        text        NOT NULL,
    args        jsonb       NOT NULL DEFAULT '{}'::jsonb,
    summary     text        NOT NULL DEFAULT '',
    status      text        NOT NULL DEFAULT 'pending',
    created_at  timestamptz NOT NULL DEFAULT now(),
    resolved_at timestamptz
);
CREATE INDEX IF NOT EXISTS ix_pending_open
    ON pending_action (tenant_id, status, created_at DESC);

-- ---------------------------------------------------------------------------
-- 召回率基线。🔴 HNSW 退化完全无声，必须定时用精确检索当 ground truth 比对。
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS recall_probe (
    id          bigserial   PRIMARY KEY,
    tenant_id   text        NOT NULL,
    kb_id       text        NOT NULL,
    probed_at   timestamptz NOT NULL DEFAULT now(),
    sample_size int         NOT NULL,
    top_k       int         NOT NULL,
    recall      double precision NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_recall_time ON recall_probe (tenant_id, probed_at DESC);

-- ---------------------------------------------------------------------------
-- 建租户分区。故意不建 DEFAULT 分区：未登记租户写入应当直接失败（fail-closed），
-- 而不是悄悄落进公共分区和别人混在一起。
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION ics_ensure_tenant(p_tenant text) RETURNS void AS $$
DECLARE
    sfx text := regexp_replace(p_tenant, '[^a-zA-Z0-9_]', '_', 'g');
BEGIN
    EXECUTE format(
        'CREATE TABLE IF NOT EXISTS document_%s PARTITION OF document FOR VALUES IN (%L)',
        sfx, p_tenant);
    EXECUTE format(
        'CREATE TABLE IF NOT EXISTS doc_chunk_%s PARTITION OF doc_chunk FOR VALUES IN (%L)',
        sfx, p_tenant);
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS ix_chunk_%s_tsv ON doc_chunk_%s USING gin (lexical_tsv)',
        sfx, sfx);
    -- 过滤列建普通索引：kb_id 取值少，命中比例低时比走向量索引更快
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS ix_chunk_%s_kb ON doc_chunk_%s (kb_id)', sfx, sfx);
    -- HNSW。召回不够优先调运行时 hnsw.ef_search（见 infra/pg.py），而不是重建索引。
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS ix_chunk_%s_vec ON doc_chunk_%s '
        'USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)',
        sfx, sfx);
END;
$$ LANGUAGE plpgsql;
