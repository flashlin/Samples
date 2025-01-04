using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlTableSource : ITableSource
{
    public SqlType SqlType { get; } = SqlType.TableSource;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_TableSource(this);
    }

    public string TableName { get; set; } = string.Empty;
    public string Alias { get; set; } = string.Empty;
    public List<ISqlExpression> Withs { get; set; } = [];

    public virtual string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write(TableName);
        WriteSqlAfterTableName(sql);
        return TableName;
    }

    protected void WriteSqlAfterTableName(IndentStringBuilder sql)
    {
        WriteAliasSql(sql);
        WriteWithsSql(sql);
    }

    private void WriteAliasSql(IndentStringBuilder sql)
    {
        if (!string.IsNullOrEmpty(Alias))
        {
            sql.Write($" AS {Alias}");
        }
    }

    private void WriteWithsSql(IndentStringBuilder sql)
    {
        if (Withs.Count > 0)
        {
            sql.Write(" WITH(");
            foreach (var with in Withs.Select((value,index)=> new {value, index}))
            {
                sql.Write($"{with.value.ToSql()}");
                if (with.index < Withs.Count - 1)
                {
                    sql.Write(", ");
                }
            }
            sql.Write(")");
        }
    }
}