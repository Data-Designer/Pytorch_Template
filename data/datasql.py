# 用于从数据库或者文件读取数据
sql_guildinfo = """
        select * from table_name 
        where ds = '{}' and guild_id in (
                -- 按最后一天id进行过滤
                select guild_id
                from table_name2
                where ds = '{}' and DATEDIFF(ds,create_date)>=14
        )
        order by guild_id
    """