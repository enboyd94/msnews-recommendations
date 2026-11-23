with raw_behaviors as (
    select *
    from read_csv(
        '../data/behaviors_train.tsv',
        delim = '\t',
        header = false,
        columns = {
            'impression_id': 'VARCHAR',
            'user_id': 'VARCHAR',
            'time': 'VARCHAR',
            'history': 'VARCHAR',
            'impressions': 'VARCHAR'
        }
    )
)

select
    impression_id,
    user_id,
    time,
    history
from raw_behaviors
where history is not null