with session_counts as (
    select
        user_id,
        count(*) as sessions_7d
    from {{ ref('raw_behaviors') }}
    group by user_id
),

reading_stats as (
    select
        i.user_id,
        count(distinct i.article_id) as articles_read_30d,
        count(distinct a.subcategory) as sport_diversity
    from {{ ref('sports_interactions') }} i
    left join {{ ref('sports_news') }} a
        on i.article_id = a.article_id
    group by i.user_id
)

select
    r.user_id,
    r.articles_read_30d,
    coalesce(s.sessions_7d, 1) as sessions_7d,
    r.sport_diversity
from reading_stats r
left join session_counts s
    on r.user_id = s.user_id