using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlCreateTableExpression : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateTable;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateTableExpression(this);
    }

    public string TableName { get; set; } = string.Empty;
    public List<ISqlExpression> Columns { get; set; } = [];
    public List<ISqlConstraint> Constraints { get; set; } = [];

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.WriteLine($"CREATE TABLE {TableName}");
        sql.WriteLine("(");
        sql.Indent++;
        for (var i = 0; i < Columns.Count; i++)
        {
            sql.Write(Columns[i].ToSql());
            if (i < Columns.Count - 1)
            {
                sql.Write(",");
            }
            sql.WriteLine();
        }
        for (var i = 0; i < Constraints.Count; i++)
        {
            sql.Write(Constraints[i].ToSql());
            if (i < Constraints.Count - 1)
            {
                sql.Write(",");
            }
            sql.WriteLine();
        }
        sql.Indent--;
        sql.WriteLine(")");
        return sql.ToString();
    }
}