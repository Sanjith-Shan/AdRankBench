-- Which crossed pairs actually move CTR relative to their own marginals.
--
-- The pipeline builds explicit second order crosses. A cross earns its bucket
-- space only when the joint cell CTR differs from what the two marginals already
-- predict. If it does not, the cross is a redundant re encoding of information a
-- linear model already has, and it costs a 100000 bucket embedding table for
-- nothing.
--
-- The null model is multiplicative in the lift, which is the natural null on a
-- rate. If value a lifts CTR by a factor of 1.3 and value b lifts it by 0.8,
-- then under no interaction the joint cell should sit at 1.3 * 0.8 = 1.04 times
-- baseline. The interaction is whatever the cell does beyond that. Measuring it
-- in log space makes a lift of 2x and a lift of 0.5x count equally, which is
-- correct, since suppression is as useful to a ranker as amplification.
--
-- The per pair score is the impression weighted mean absolute log residual over
-- cells large enough to measure. Weighting by impressions is what stops a pair
-- from scoring highly on a handful of thin cells that happen to be extreme.
WITH base AS (
    SELECT avg(label)::DOUBLE AS base_ctr, count(*) AS total_rows
    FROM impressions
),
pairs AS (
{pair_union}
),
marginal_a AS (
    SELECT pair, a_value, count(*) AS n_a, avg(label)::DOUBLE AS ctr_a
    FROM pairs GROUP BY pair, a_value
),
marginal_b AS (
    SELECT pair, b_value, count(*) AS n_b, avg(label)::DOUBLE AS ctr_b
    FROM pairs GROUP BY pair, b_value
),
cells AS (
    SELECT pair, a_value, b_value, count(*) AS n, avg(label)::DOUBLE AS ctr
    FROM pairs GROUP BY pair, a_value, b_value
),
scored AS (
    SELECT
        c.pair,
        c.a_value,
        c.b_value,
        c.n,
        c.ctr,
        b.base_ctr,
        ma.ctr_a,
        mb.ctr_b,
        greatest(
            least(b.base_ctr * (ma.ctr_a / b.base_ctr) * (mb.ctr_b / b.base_ctr), 1 - {eps}),
            {eps}
        ) AS expected_ctr
    FROM cells c
    JOIN marginal_a ma ON ma.pair = c.pair AND ma.a_value = c.a_value
    JOIN marginal_b mb ON mb.pair = c.pair AND mb.b_value = c.b_value
    CROSS JOIN base b
    WHERE c.n >= {min_cell}
),
residuals AS (
    SELECT
        pair,
        n,
        ctr,
        expected_ctr,
        ln(greatest(ctr, {eps}) / expected_ctr) AS log_residual
    FROM scored
)
SELECT
    pair,
    count(*)                                        AS measurable_cells,
    sum(n)                                          AS impressions_covered,
    sum(n * abs(log_residual)) / sum(n)             AS weighted_abs_log_lift,
    exp(sum(n * abs(log_residual)) / sum(n))        AS mean_lift_factor,
    max(abs(log_residual))                          AS max_abs_log_lift,
    coalesce(sum(n) FILTER (WHERE abs(log_residual) > ln(1.5)), 0) / sum(n)::DOUBLE
                                                    AS share_beyond_1_5x
FROM residuals
GROUP BY pair
ORDER BY weighted_abs_log_lift DESC
