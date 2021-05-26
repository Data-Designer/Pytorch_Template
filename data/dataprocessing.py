# 数据加载和预处理，传入dataset
from rslib.utils.datadownload import datadownload

def sqlinfo(sql,ds,ds_end):
    '''
    数据查询，在这里传入dataset
    :param sql:
    :param ds:
    :return:
    '''
    df = datadownload(sql.format(ds,ds_end))
    print(df.shape)
    return df