-- The individual joint cells with the largest interaction residual.
--
-- The per pair summary says which crosses are worth building. This one shows the
-- specific conjunctions driving that, so the finding can be inspected rather
-- than taken on trust. Every row carries its impression count, and the cell
-- floor is applied before ranking so a two impression cell can never appear.
WITH base AS (
    SELECT avg(label)::DOUBLE AS base_ctr FROM impressions
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
)
SELECT
    c.pair,
    c.a_value,
    c.b_value,
    c.n                                             AS impressions,
    c.ctr                                           AS observed_ctr,
    ma.ctr_a                                        AS marginal_ctr_a,
    mb.ctr_b                                        AS marginal_ctr_b,
    b.base_ctr * (ma.ctr_a / b.base_ctr) * (mb.ctr_b / b.base_ctr) AS expected_ctr,
    c.ctr / greatest(
        b.base_ctr * (ma.ctr_a / b.base_ctr) * (mb.ctr_b / b.base_ctr), {eps}
    )                                               AS interaction_lift
FROM cells c
JOIN marginal_a ma ON ma.pair = c.pair AND ma.a_value = c.a_value
JOIN marginal_b mb ON mb.pair = c.pair AND mb.b_value = c.b_value
CROSS JOIN base b
WHERE c.n >= {min_cell}
ORDER BY abs(ln(greatest(c.ctr, {eps}) / greatest(
    b.base_ctr * (ma.ctr_a / b.base_ctr) * (mb.ctr_b / b.base_ctr), {eps}
))) DESC
LIMIT {limit}
