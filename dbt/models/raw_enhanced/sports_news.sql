
SELECT *
FROM {{ref('raw_news')}}
WHERE 1=1
AND category = 'sports'