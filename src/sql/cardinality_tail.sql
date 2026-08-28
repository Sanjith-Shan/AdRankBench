-- Long tail cardinality per sparse field.
--
-- This is the query that turns the hash encoding decision from a convention
-- into a measurement. One hot encoding is only unreasonable if the fields are
-- actually high cardinality with a long tail, and that is a fact about the data,
-- not an assumption. The numbers below say how many distinct values each field
-- takes, how concentrated the traffic is in the head, where the tail begins, and
-- how much traffic sits below the rare token cutoff the pipeline already uses.
--
-- `values_for_90pct` is the tail boundary. It is the number of distinct values,
-- ranked by volume, needed to cover 90 percent of impressions. When that number
-- is a small fraction of `distinct_values`, almost every distinct value in the
-- field is a near singleton, a one hot column for it would be almost always
-- zero, and an embedding row for it would never get enough gradient to train.
-- That is precisely the case hashing is for.
WITH long AS (
    SELECT field, value
    FROM impressions
    UNPIVOT ( value FOR field IN ({cat_columns}) )
),
counts AS (
    SELECT field, value, count(*) AS n
    FROM long
    GROUP BY field, value
),
ranked AS (
    SELECT
        field,
        value,
        n,
        row_number() OVER (
            PARTITION BY field ORDER BY n DESC, value
        ) AS rank_in_field,
        sum(n) OVER (PARTITION BY field) AS field_rows,
        sum(n) OVER (
            PARTITION BY field ORDER BY n DESC, value
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_rows
    FROM counts
)
SELECT
    field,
    any_value(field_rows)                                          AS impressions,
    count(*)                                                       AS distinct_values,
    count(*) / any_value(field_rows)::DOUBLE                       AS distinct_per_impression,
    coalesce(sum(n) FILTER (WHERE rank_in_field <= 10), 0)
        / any_value(field_rows)::DOUBLE                            AS top10_coverage,
    coalesce(sum(n) FILTER (WHERE rank_in_field <= 100), 0)
        / any_value(field_rows)::DOUBLE                            AS top100_coverage,
    coalesce(sum(n) FILTER (WHERE rank_in_field <= 1000), 0)
        / any_value(field_rows)::DOUBLE                            AS top1000_coverage,
    min(rank_in_field) FILTER (
        WHERE cumulative_rows >= 0.90 * field_rows
    )                                                              AS values_for_90pct,
    count(*) FILTER (WHERE n < {min_count})                        AS values_below_min_count,
    coalesce(sum(n) FILTER (WHERE n < {min_count}), 0)
        / any_value(field_rows)::DOUBLE                            AS rare_traffic_share,
    count(*) FILTER (WHERE n = 1)                                  AS singleton_values
FROM ranked
GROUP BY field
ORDER BY distinct_values DESC
