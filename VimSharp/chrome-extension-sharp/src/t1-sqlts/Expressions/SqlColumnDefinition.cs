namespace T1.SqlSharp.Expressions;

public class SqlColumnDefinition : ISqlExpression
{
    public SqlType SqlType => SqlType.ColumnDefinition;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_ColumnDefinition(this);
    }

    public string ColumnName { get; set; } = string.Empty;
    public string DataType { get; set; } = string.Empty;
    public SqlDataSize? DataSize { get; set; }
    public SqlIdentity Identity { get; set; } = SqlIdentity.Default;
    public bool IsNullable { get; set; } = true;
    public bool NotForReplication { get; set; }
    public List<ISqlExpression> Constraints { get; set; } = [];
    public bool IsPrimaryKey { get; set; }

    public string ToSql()
    {
        var sql = new T1.Standard.IO.IndentStringBuilder();
        sql.Write(ColumnName);
        sql.Write(" ");
        sql.Write(DataType);
        if(DataSize != null)
        {
            sql.Write(DataSize.ToSql());
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

        if (IsPrimaryKey)
        {
            sql.Write(" PRIMARY KEY");
        }
        if (Constraints.Count > 0)
        {
            sql.Write(" ");
            sql.Write(string.Join(", ", Constraints.Select(c => c.ToSql())));
        }
        return sql.ToString();
    }
}