-- Missingness in the dense features, and whether it correlates with the label.
--
-- The pipeline fills missing dense values with zero and appends a binary
-- is_missing indicator per column, which doubles the dense feature width from 13
-- to 26. That is only worth paying for if missingness carries signal. If a
-- feature is missing at random with respect to the click, the indicator is 13
-- columns of noise and the fill value is all that matters. If it is not, the
-- indicator is free signal that a plain zero fill throws away.
--
-- This query measures it directly. For each dense column it reports the missing
-- rate, the CTR on the rows where the column is missing, the CTR on the rows
-- where it is present, and a two proportion z statistic for the difference. The
-- z statistic uses the pooled proportion under the null, so a large absolute
-- value means the gap is far too big to be volume noise. Both sample sizes are
-- reported next to it, because a z statistic without its n is not a finding.
WITH long AS (
    SELECT field, value, label
    FROM impressions
    UNPIVOT INCLUDE NULLS ( value FOR field IN ({dense_columns}) )
),
counts AS (
    SELECT
        field,
        count(*)                                          AS impressions,
        count(*) FILTER (WHERE value IS NULL)             AS n_missing,
        sum(label) FILTER (WHERE value IS NULL)           AS clicks_missing,
        count(*) FILTER (WHERE value IS NOT NULL)         AS n_present,
        sum(label) FILTER (WHERE value IS NOT NULL)       AS clicks_present
    FROM long
    GROUP BY field
),
rates AS (
    SELECT
        field,
        impressions,
        n_missing,
        n_present,
        n_missing / impressions::DOUBLE                            AS missing_rate,
        clicks_missing / nullif(n_missing, 0)::DOUBLE              AS ctr_missing,
        clicks_present / nullif(n_present, 0)::DOUBLE              AS ctr_present,
        (clicks_missing + clicks_present) / impressions::DOUBLE    AS ctr_pooled
    FROM counts
)
SELECT
    field,
    impressions,
    n_missing,
    n_present,
    missing_rate,
    ctr_missing,
    ctr_present,
    ctr_missing - ctr_present AS ctr_delta,
    CASE
        WHEN n_missing = 0 OR n_present = 0 THEN NULL
        WHEN ctr_pooled <= 0 OR ctr_pooled >= 1 THEN NULL
        ELSE (ctr_missing - ctr_present)
             / sqrt(
                 ctr_pooled * (1 - ctr_pooled)
                 * (1.0 / n_missing + 1.0 / n_present)
               )
    END AS z_statistic
FROM rates
ORDER BY abs(coalesce(ctr_missing - ctr_present, 0)) DESC
