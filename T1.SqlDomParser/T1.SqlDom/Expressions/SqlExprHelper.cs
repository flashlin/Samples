namespace T1.SqlDom.Expressions;

public static class SqlExprHelper
{
    public static string ToSqlString<TSqlExpr>(this ICollection<TSqlExpr> sqlExprList, string delimiter)
        where TSqlExpr: SqlExpr
    {
        var sqlExprStrList = sqlExprList.Select(x => x.ToSqlString());
        return string.Join(delimiter, sqlExprStrList);
    }
}