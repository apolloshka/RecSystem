import random
import sys
import math

import joblib
import pandas as pd

from src.db.clickhouse_client import get_client
from src.recommenders.common import (
    build_group_popularity,
    build_group_to_users,
    build_profile_members,
    build_user_to_groups,
    extract_feature_dict_for_profile_candidate,
    get_item_based_scores_for_profile,
    get_user_based_scores_for_profile,
    sample_leave_one_out_targets,
)
from src.recommenders.config import (
    RANDOM_SEED,
    TOP_K_SIMILAR_USERS,
    MIN_GROUP_SIZE,
    MIN_USER_SIMILARITY,
    MIN_COMMON_GROUPS,
    MAX_GROUP_POPULARITY,
    MIN_ITEM_GROUP_SIZE,
    MIN_ITEM_SIMILARITY,
    MIN_ITEM_SUPPORT,
    MAX_ITEM_CANDIDATES,
    ENABLE_ITEM_BASED,
    TOP_USER_BASED_CANDIDATES,
    TOP_ITEM_BASED_CANDIDATES,
    TOP_BASELINE_BLACKLIST,
    MAX_RECOMMENDED_GROUP_POPULARITY,
    MIN_CF_SCORE,
    MIN_MAX_GROUP_SIMILARITY,
    TOP_K_RECOMMENDATIONS,
    MODEL_BUNDLE_PATH,
    TEST_USERS_PATH,
    TRAIN_USERS_PATH,
    RANKING_EVAL_CSV_PATH,
    RANKING_GATE_MIN_RECALL_AT_10,
    RANKING_EVAL_MAX_USERS,
    RANKING_EVAL_PROGRESS_EVERY,
)


def dcg_at_k(rank: int, k: int) -> float:
    if rank <= 0 or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def main():
    print("=== Ranking evaluation on held-out users ===")
    random.seed(RANDOM_SEED)

    client = get_client()
    model_bundle = joblib.load(MODEL_BUNDLE_PATH)
    pipeline = model_bundle["pipeline"]
    feature_names = model_bundle["feature_names"]
    test_users = set(joblib.load(TEST_USERS_PATH))
    train_users = set(joblib.load(TRAIN_USERS_PATH))

    rows = client.query("SELECT user_id, group_id FROM user_groups").result_rows
    all_user_to_groups = build_user_to_groups(rows)
    train_user_to_groups = {u: g for u, g in all_user_to_groups.items() if u in train_users}
    train_group_to_users = build_group_to_users(train_user_to_groups)
    train_group_popularity = build_group_popularity(train_group_to_users)

    my_groups = {
        int(row[0])
        for row in client.query("SELECT group_id FROM my_groups").result_rows
    }
    baseline_blacklist = [
        gid
        for gid, pop in sorted(
            train_group_popularity.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        if gid not in my_groups and pop >= MIN_GROUP_SIZE
    ]

    baseline_blacklist = set(baseline_blacklist[:TOP_BASELINE_BLACKLIST])

    eval_rows = []
    evaluated_targets = 0
    hits_at_10 = 0
    ndcg_at_10_sum = 0.0
    mrr_sum = 0.0

    eligible_test_users = sorted([
        uid for uid in test_users if uid in all_user_to_groups and len(all_user_to_groups[uid]) >= 5
    ])[:RANKING_EVAL_MAX_USERS]
    print(f"Eligible test users: {len(eligible_test_users)}")
    print("Starting evaluation loop...")

    for idx, user_id in enumerate(eligible_test_users, 1):
        real_groups = set(all_user_to_groups[user_id])
        targets = sample_leave_one_out_targets(
            real_groups=real_groups,
            max_positives_per_user=2,
            random_seed=RANDOM_SEED + idx,
        )

        for target_group in targets:
            profile_groups = real_groups - {target_group}
            if len(profile_groups) < 3:
                continue

            user_based_scores = get_user_based_scores_for_profile(
                profile_groups=profile_groups,
                all_user_groups=train_user_to_groups,
                group_popularity=train_group_popularity,
                top_k_users=TOP_K_SIMILAR_USERS,
                min_group_size=MIN_GROUP_SIZE,
                min_similarity=MIN_USER_SIMILARITY,
                min_common_groups=MIN_COMMON_GROUPS,
                max_group_popularity=MAX_GROUP_POPULARITY,
                target_user_id=user_id,
            )

            if ENABLE_ITEM_BASED:
                item_based_scores = get_item_based_scores_for_profile(
                    profile_groups=profile_groups,
                    group_to_users=train_group_to_users,
                    user_to_groups=train_user_to_groups,
                    min_item_group_size=MIN_ITEM_GROUP_SIZE,
                    min_item_similarity=MIN_ITEM_SIMILARITY,
                    min_item_support=MIN_ITEM_SUPPORT,
                    max_item_candidates=MAX_ITEM_CANDIDATES,
                    target_user_id=user_id,
                )
            else:
                item_based_scores = {}

            profile_members = build_profile_members(
                profile_groups=profile_groups,
                group_to_users=train_group_to_users,
                target_user_id=user_id,
            )

            user_based_candidates = [
                gid
                for gid, _ in sorted(user_based_scores.items(), key=lambda x: x[1], reverse=True)
                if gid not in profile_groups
            ][:TOP_USER_BASED_CANDIDATES]

            item_based_candidates = [
                gid
                for gid, _ in sorted(item_based_scores.items(), key=lambda x: x[1], reverse=True)
                if gid not in profile_groups
            ][:TOP_ITEM_BASED_CANDIDATES]

            candidates = set(user_based_candidates) | set(item_based_candidates)
            candidates = candidates - profile_groups - baseline_blacklist
            candidates = {
                gid for gid in candidates if train_group_popularity.get(gid, 0) <= MAX_RECOMMENDED_GROUP_POPULARITY
            }

            filtered_candidates = []
            for gid in candidates:
                feature_dict = extract_feature_dict_for_profile_candidate(
                    profile_groups=profile_groups,
                    candidate_group=gid,
                    user_based_scores=user_based_scores,
                    item_based_scores=item_based_scores,
                    group_to_users=train_group_to_users,
                    profile_members=profile_members,
                    target_user_id=user_id,
                )
                has_cf_score = (
                    feature_dict["user_based_score"] >= MIN_CF_SCORE
                    or feature_dict["item_based_score"] >= MIN_CF_SCORE
                )
                has_group_similarity = feature_dict["max_group_similarity"] >= MIN_MAX_GROUP_SIMILARITY
                if has_cf_score and has_group_similarity:
                    filtered_candidates.append(gid)

            filtered_candidates = list(set(filtered_candidates))
            target_in_candidates = 1 if target_group in filtered_candidates else 0

            if not filtered_candidates:
                rank = 10**9

                hit10 = 0.0
                ndcg10 = 0.0
                mrr = 0.0

                evaluated_targets += 1
                hits_at_10 += hit10
                ndcg_at_10_sum += ndcg10
                mrr_sum += mrr

                eval_rows.append(
                    {
                        "user_id": user_id,
                        "target_group": target_group,
                        "rank": rank,
                        "hr@10": hit10,
                        "ndcg@10": ndcg10,
                        "mrr": mrr,
                        "candidates_count": len(ranked_groups),
                        "target_in_candidates": target_in_candidates,
                    }
                )

                continue

            feature_dicts = []
            for gid in filtered_candidates:
                feature_dict = extract_feature_dict_for_profile_candidate(
                    profile_groups=profile_groups,
                    candidate_group=gid,
                    user_based_scores=user_based_scores,
                    item_based_scores=item_based_scores,
                    group_to_users=train_group_to_users,
                    profile_members=profile_members,
                    target_user_id=user_id,
                )
                feature_dict = {f: feature_dict.get(f, 0.0) for f in feature_names}
                feature_dicts.append((gid, feature_dict))

            features_df = pd.DataFrame([fd for _, fd in feature_dicts], columns=feature_names)
            raw_scores = pipeline.predict_proba(features_df)[:, 1]
            pred_rows = list(zip([gid for gid, _ in feature_dicts], raw_scores))

            pred_rows.sort(key=lambda x: x[1], reverse=True)
            ranked_groups = [gid for gid, _ in pred_rows]
            rank = ranked_groups.index(target_group) + 1 if target_group in ranked_groups else 10**9

            hit10 = 1.0 if rank <= 10 else 0.0
            ndcg10 = dcg_at_k(rank, 10)
            mrr = 1.0 / rank if rank < 10**9 else 0.0

            evaluated_targets += 1
            hits_at_10 += hit10
            ndcg_at_10_sum += ndcg10
            mrr_sum += mrr

            eval_rows.append(
                {
                    "user_id": user_id,
                    "target_group": target_group,
                    "rank": rank,
                    "hr@10": hit10,
                    "ndcg@10": ndcg10,
                    "mrr": mrr,
                    "candidates_count": len(ranked_groups),
                }
            )

        if idx % RANKING_EVAL_PROGRESS_EVERY == 0 or idx == len(eligible_test_users):
            current_recall = (hits_at_10 / evaluated_targets) if evaluated_targets else 0.0
            print(
                f"[PROGRESS] users={idx}/{len(eligible_test_users)} | "
                f"targets={evaluated_targets} | recall@10={current_recall:.4f}"
            )

    if not eval_rows:
        raise ValueError("No evaluation rows produced. Check data volume/filters.")

    df_eval = pd.DataFrame(eval_rows)
    df_eval.to_csv(RANKING_EVAL_CSV_PATH, index=False)

    recall_at_10 = hits_at_10 / evaluated_targets
    ndcg_at_10 = ndcg_at_10_sum / evaluated_targets
    mean_mrr = mrr_sum / evaluated_targets

    print("\n=== Ranking metrics ===")
    print(f"Evaluated targets: {evaluated_targets}")
    print(f"Recall@10: {recall_at_10:.4f}")
    print(f"NDCG@10: {ndcg_at_10:.4f}")
    print(f"MRR: {mean_mrr:.4f}")
    print(f"Saved details to: {RANKING_EVAL_CSV_PATH}")

    if recall_at_10 < RANKING_GATE_MIN_RECALL_AT_10:
        print(
            f"[GATE FAILED] Recall@10={recall_at_10:.4f} < "
            f"{RANKING_GATE_MIN_RECALL_AT_10:.4f}"
        )
        sys.exit(1)

    print("[GATE PASSED] Ranking quality is acceptable.")


if __name__ == "__main__":
    main()
