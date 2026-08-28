-- Rank the sparse fields by the variance of their training value frequencies.
--
-- This reproduces the selection rule inside `src.features.crosses.CrossGenerator`
-- in SQL. The generator computes each field's normalized value frequency
-- distribution on the training split, takes the population variance of that
-- distribution, ranks descending, and crosses the top k. A high variance means
-- the field is dominated by a few heavy values, which is what makes its
-- conjunctions dense enough to learn from. A near uniform field has variance
-- close to zero and its cells are all thin.
--
-- Running the rule as a query rather than trusting the Python is the point. It
-- means the analysis can show which fields the pipeline crosses and, separately,
-- whether those fields turn out to be the ones that actually carry interaction.
WITH train_rows AS (
    SELECT * FROM impressions WHERE row_id < {n_train}
),
long AS (
    SELECT field, value
    FROM train_rows
    UNPIVOT ( value FOR field IN ({cat_columns}) )
),
counts AS (
    SELECT field, value, count(*) AS n
    FROM long
    GROUP BY field, value
),
frequencies AS (
    SELECT field, value, n, n / {n_train}.0 AS frequency
    FROM counts
)
SELECT
    field,
    count(*)               AS distinct_values,
    var_pop(frequency)     AS frequency_variance,
    max(frequency)         AS top_value_share,
    sum(n)                 AS train_impressions
FROM frequencies
GROUP BY field
ORDER BY frequency_variance DESC, field ASC
