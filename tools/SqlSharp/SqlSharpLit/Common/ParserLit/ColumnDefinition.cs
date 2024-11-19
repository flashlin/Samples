namespace SqlSharpLit.Common.ParserLit;

public class ColumnDefinition : ISqlExpression
{
    public SqlType SqlType => SqlType.ColumnDefinition;
    public string ColumnName { get; set; } = string.Empty;
    public string DataType { get; set; } = string.Empty;
    public int Size { get; set; }
    public int Scale { get; set; }
    public SqlIdentity Identity { get; set; } = SqlIdentity.Default;
    public bool IsNullable { get; set; }
    public bool NotForReplication { get; set; }
    public List<ISqlConstraint> Constraints { get; set; } = [];

    public string ToSql()
    {
        var sql = new T1.Standard.IO.IndentStringBuilder();
        sql.Write(ColumnName);
        sql.Write(" ");
        sql.Write(DataType);
        if (Size > 0)
        {
            sql.Write("(");
            sql.Write($"({Size}");
            if (Scale > 0)
            {
                sql.Write($", {Scale}");
            }
            sql.Write(")");
        }
        if (Identity != SqlIdentity.Default)
        {
            sql.Write(" ");
            sql.Write(Identity.ToSql());
        }
        if (IsNullable)
        {
            sql.Write(" NULL");
        }
        else
        {
            sql.Write(" NOT NULL");
        }
        if (NotForReplication)
        {
            sql.Write(" NOT FOR REPLICATION");
        }
        if (Constraints.Count > 0)
        {
            sql.Write(" ");
            sql.Write(string.Join(", ", Constraints.Select(c => c.ToSql())));
        }
        return sql.ToString();
    }
}