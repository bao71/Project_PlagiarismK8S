CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_name   TEXT        NOT NULL,
    subject_id  TEXT        NOT NULL,
    file_path   TEXT        NOT NULL,
    minhash     INTEGER[],            
    created_at  TIMESTAMP   NOT NULL DEFAULT NOW()
);

-- Index tìm kiếm theo môn học (dùng khi lọc thô MinHash)
CREATE INDEX IF NOT EXISTS idx_documents_subject_id ON documents(subject_id);

-- Index tìm kiếm theo tên file
CREATE INDEX IF NOT EXISTS idx_documents_file_name ON documents(file_name);
