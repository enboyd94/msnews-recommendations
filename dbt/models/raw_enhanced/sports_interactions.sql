with latest_user_history as (
    select
        user_id,
        history
    from {{ ref('raw_behaviors') }}
    qualify row_number() over (partition by user_id order by time desc) = 1
),

exploded_clicks as (
    select distinct
        user_id,
        unnest(string_split(history, ' ')) as article_id
    from latest_user_history
),

sports_clicks as (
    select
        c.user_id,
        c.article_id
    from exploded_clicks c
    inner join {{ ref('sports_news') }} a
        on c.article_id = a.article_id
),

active_users as (
    select user_id
    from sports_clicks
    group by user_id
    having count(*) >= 5
)

select
    s.user_id,
    s.article_id
from sports_clicks s
inner join active_users u
    on s.user_id = u.user_id