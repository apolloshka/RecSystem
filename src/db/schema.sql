-- ============================================
-- VK Recommendation Pipeline
-- Схема базы данных ClickHouse
-- ============================================

-- 1. ОСНОВНЫЕ ТАБЛИЦЫ
-- ============================================

-- Мои группы (целевой пользователь)
CREATE TABLE IF NOT EXISTS my_groups (
    group_id UInt64,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (group_id, created_at);

-- Связи "пользователь → группа"
CREATE TABLE IF NOT EXISTS user_groups (
    user_id UInt64,
    group_id UInt64,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (user_id, group_id, created_at);

-- Связи "группа → участник"
CREATE TABLE IF NOT EXISTS group_members (
    source_group_id UInt64,
    member_id UInt64,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (source_group_id, member_id, created_at);


-- 2. ТАБЛИЦЫ РЕКОМЕНДАЦИЙ
-- ============================================

-- User-based рекомендации
CREATE TABLE IF NOT EXISTS user_based_recommendations (
    recommended_group_id UInt64,
    group_name String,
    score Float64,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (score, created_at);

-- Item-based рекомендации
CREATE TABLE IF NOT EXISTS item_based_recommendations (
    recommended_group_id UInt64,
    group_name String,
    score Float64,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (score, created_at);

-- Baseline (популярные группы)
CREATE TABLE IF NOT EXISTS baseline_recommendations (
    group_id UInt64,
    group_name String,
    members_count UInt64,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (members_count, created_at);


-- 3. ТАБЛИЦЫ ПОХОЖЕСТИ
-- ============================================

-- Похожие пользователи
CREATE TABLE IF NOT EXISTS user_similarity (
    user_id UInt64,
    similarity Float64,
    common_groups_count UInt64,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (similarity, created_at);

-- Похожие группы
CREATE TABLE IF NOT EXISTS group_similarity (
    source_group_id UInt64,
    target_group_id UInt64,
    similarity Float64,
    common_users_count UInt64,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (source_group_id, similarity, created_at);


-- 4. ML ДАТАСЕТ
-- ============================================

-- Обучающий датасет (базовая версия с 4 признаками)
CREATE TABLE IF NOT EXISTS ml_dataset (
    user_id UInt64,
    candidate_group_id UInt64,
    label UInt8,
    group_popularity Float64,
    log_group_popularity Float64,
    user_based_score Float64,
    item_based_score Float64,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (user_id, candidate_group_id, created_at);


-- 5. КЭШ НАЗВАНИЙ ГРУПП
-- ============================================

-- Кэш для названий групп (чтобы не дёргать VK API)
CREATE TABLE IF NOT EXISTS groups (
    group_id UInt64,
    name String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY group_id;


-- 6. ОЧИСТКА ВСЕХ ТАБЛИЦ (опционально)
-- ============================================
-- TRUNCATE TABLE my_groups;
-- TRUNCATE TABLE user_groups;
-- TRUNCATE TABLE group_members;
-- TRUNCATE TABLE user_based_recommendations;
-- TRUNCATE TABLE item_based_recommendations;
-- TRUNCATE TABLE baseline_recommendations;
-- TRUNCATE TABLE user_similarity;
-- TRUNCATE TABLE group_similarity;
-- TRUNCATE TABLE ml_dataset;
-- TRUNCATE TABLE groups;


-- 7. УДАЛЕНИЕ ВСЕХ ТАБЛИЦ (опционально)
-- ============================================
-- DROP TABLE IF EXISTS my_groups;
-- DROP TABLE IF EXISTS user_groups;
-- DROP TABLE IF EXISTS group_members;
-- DROP TABLE IF EXISTS user_based_recommendations;
-- DROP TABLE IF EXISTS item_based_recommendations;
-- DROP TABLE IF EXISTS baseline_recommendations;
-- DROP TABLE IF EXISTS user_similarity;
-- DROP TABLE IF EXISTS group_similarity;
-- DROP TABLE IF EXISTS ml_dataset;
-- DROP TABLE IF EXISTS groups;