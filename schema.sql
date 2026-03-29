CREATE DATABASE IF NOT EXISTS vk_diploma;

CREATE TABLE IF NOT EXISTS vk_diploma.my_groups
(
    group_id UInt64,
    loaded_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY group_id;

CREATE TABLE IF NOT EXISTS vk_diploma.user_groups
(
    user_id UInt64,
    group_id UInt64,
    loaded_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (user_id, group_id);

CREATE TABLE IF NOT EXISTS vk_diploma.group_members
(
    source_group_id UInt64,
    member_id UInt64,
    loaded_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (source_group_id, member_id);