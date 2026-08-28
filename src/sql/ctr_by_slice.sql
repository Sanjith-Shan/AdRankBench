-- CTR by categorical feature slice, with Wilson score confidence intervals.
--
-- The naive version of this query is a group by with an average, and it is
-- misleading. A slice with twelve impressions and four clicks reports a 33
-- percent CTR and sorts to the top of every list, and it means nothing. The
-- Wilson score interval is the fix. It is the inversion of the score test for a
-- binomial proportion, it does not collapse at zero clicks the way a normal
-- approximation does, and its width falls with the square root of the volume,
-- so a low volume slice is visibly uncertain rather than quietly wrong.
--
-- A slice counts as a real finding only when its interval excludes the dataset
-- baseline CTR entirely. That is the whole point of carrying the interval.
WITH baseline AS (
    SELECT
        avg(label)::DOUBLE AS base_ctr,
        count(*)           AS total_rows
    FROM impressions
),
long AS (
    SELECT field, value, label
    FROM impressions
    UNPIVOT ( value FOR field IN ({cat_columns}) )
),
slices AS (
    SELECT
        field,
        value,
        count(*)           AS impressions,
        sum(label)         AS clicks,
        avg(label)::DOUBLE AS ctr
    FROM long
    GROUP BY field, value
    HAVING count(*) >= {min_impressions}
),
scored AS (
    SELECT
        s.field,
        s.value,
        s.impressions,
        s.clicks,
        s.ctr,
        b.base_ctr,
        -- Wilson score interval at z = {z}.
        (s.ctr + {z} * {z} / (2 * s.impressions))
            / (1 + {z} * {z} / s.impressions) AS wilson_center,
        {z} * sqrt(
            s.ctr * (1 - s.ctr) / s.impressions
            + {z} * {z} / (4 * s.impressions * s.impressions)
        ) / (1 + {z} * {z} / s.impressions)   AS wilson_margin
    FROM slices s
    CROSS JOIN baseline b
)
SELECT
    field,
    value,
    impressions,
    clicks,
    ctr,
    base_ctr,
    ctr / base_ctr                            AS ctr_lift,
    wilson_center - wilson_margin             AS ctr_low,
    wilson_center + wilson_margin             AS ctr_high,
    CASE
        WHEN wilson_center - wilson_margin > base_ctr THEN true
        WHEN wilson_center + wilson_margin < base_ctr THEN true
        ELSE false
    END                                       AS separated_from_baseline
FROM scored
ORDER BY abs(ctr - base_ctr) DESC, impressions DESC
LIMIT {limit}
