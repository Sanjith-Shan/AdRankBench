-- CTR drift across the temporal split, and the same measure on random folds.
--
-- The project splits by position rather than at random, on the argument that a
-- random split leaks future information. That argument only has force if the
-- data actually changes over time. If CTR and the feature distribution were
-- stationary, a random split would leak nothing that matters and the temporal
-- split would be pure cost.
--
-- This query measures the non stationarity directly. It reports CTR for the
-- three positional splits, CTR for each equal sized block of the file in order,
-- and CTR for the same number of equal sized random folds. The random folds are
-- the control. Their CTR spread is what pure sampling noise looks like at this
-- volume, so any spread the ordered blocks show above that is real drift, and
-- real drift is exactly what a random split would hide from the test set.
WITH ordered AS (
    SELECT
        row_id,
        label,
        ntile({n_buckets}) OVER (ORDER BY row_id)          AS time_block,
        ntile({n_buckets}) OVER (ORDER BY hash(row_id))    AS random_fold
    FROM impressions
),
by_split AS (
    SELECT
        'temporal split' AS grain,
        CASE
            WHEN row_id < {n_train}            THEN 'train'
            WHEN row_id < {n_train_val}        THEN 'val'
            ELSE 'test'
        END AS bucket,
        CASE
            WHEN row_id < {n_train}            THEN 1
            WHEN row_id < {n_train_val}        THEN 2
            ELSE 3
        END AS bucket_order,
        label
    FROM ordered
),
by_block AS (
    SELECT 'time block' AS grain, 'block ' || lpad(time_block::VARCHAR, 2, '0') AS bucket,
           time_block AS bucket_order, label
    FROM ordered
),
by_random AS (
    SELECT 'random fold' AS grain, 'fold ' || lpad(random_fold::VARCHAR, 2, '0') AS bucket,
           random_fold AS bucket_order, label
    FROM ordered
),
combined AS (
    SELECT * FROM by_split
    UNION ALL SELECT * FROM by_block
    UNION ALL SELECT * FROM by_random
)
SELECT
    grain,
    bucket,
    any_value(bucket_order) AS bucket_order,
    count(*)                AS impressions,
    sum(label)              AS clicks,
    avg(label)::DOUBLE      AS ctr
FROM combined
GROUP BY grain, bucket
ORDER BY grain, bucket_order
