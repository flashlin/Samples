from utils.tsql_tokenizr import tsql_encode, tsql_decode


def test_select_1():
    sql1 = "SELECT tb1.id AS id, 'flash' AS name FROM [dbo].[myUser] AS tb1 WITH(NOLOCK)"
    values = tsql_encode(sql1)
    sql2 = tsql_decode(values)
    assert sql2 == sql1