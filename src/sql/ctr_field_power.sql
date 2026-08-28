-- How much CTR signal each sparse field carries, per field.
--
-- The slice level query answers which individual values move CTR. This one
-- rolls that up and answers which fields are worth modelling at all. The
-- statistic is the impression weighted mean absolute deviation of a slice's CTR
-- from the dataset baseline, restricted to slices large enough to measure.
-- Weighting by impressions is what keeps a field full of tiny noisy slices from
-- outranking a field with a few large, genuinely predictive ones.
--
-- `covered_share` is the fraction of all impressions that fall into slices big
-- enough to clear the volume floor. A field with high power over 3 percent of
-- traffic is a different proposition from one with the same power over 90
-- percent, so the two numbers have to be read together.
WITH baseline AS (
    SELECT avg(label)::DOUBLE AS base_ctr, count(*) AS total_rows
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
        avg(label)::DOUBLE AS ctr
    FROM long
    GROUP BY field, value
),
field_totals AS (
    SELECT field, sum(impressions) AS field_rows, count(*) AS distinct_values
    FROM slices
    GROUP BY field
),
-- The baseline is attached per row before aggregation, because an aggregate
-- cannot be nested inside another aggregate's argument.
with_baseline AS (
    SELECT
        s.field,
        s.impressions,
        s.ctr,
        b.base_ctr,
        t.field_rows,
        t.distinct_values
    FROM slices s
    JOIN field_totals t ON t.field = s.field
    CROSS JOIN baseline b
)
SELECT
    field,
    any_value(distinct_values)                                     AS distinct_values,
    count(*) FILTER (WHERE impressions >= {min_impressions})       AS measurable_slices,
    coalesce(sum(impressions) FILTER (WHERE impressions >= {min_impressions}), 0)
        AS measurable_impressions,
    coalesce(sum(impressions) FILTER (WHERE impressions >= {min_impressions}), 0)
        / any_value(field_rows)::DOUBLE                            AS covered_share,
    coalesce(
        sum(impressions * abs(ctr - base_ctr))
            FILTER (WHERE impressions >= {min_impressions})
        / nullif(sum(impressions) FILTER (WHERE impressions >= {min_impressions}), 0),
        0.0
    )                                                              AS weighted_ctr_deviation
FROM with_baseline
GROUP BY field
ORDER BY weighted_ctr_deviation DESC
