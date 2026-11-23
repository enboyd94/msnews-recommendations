with raw_news as (
    select *
    from read_csv(
        '../data/news_train.tsv',
        delim = '\t',
        header = false,
        quote = '',
        columns = {
            'article_id': 'VARCHAR',
            'category': 'VARCHAR',
            'subcategory': 'VARCHAR',
            'title': 'VARCHAR',
            'abstract': 'VARCHAR',
            'url': 'VARCHAR',
            'title_entities': 'VARCHAR',
            'abstract_entities': 'VARCHAR'
        }
    )
)

select distinct
    article_id,
    category,
    subcategory,
    title,
    coalesce(abstract, '') as abstract
from raw_news
where title is not null