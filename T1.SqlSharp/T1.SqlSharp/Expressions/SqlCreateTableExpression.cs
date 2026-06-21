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
    public string OnFileGroup { get; set; } = string.Empty;
    public string TextImageOn { get; set; } = string.Empty;
    public string Period { get; set; } = string.Empty;
    public List<string> WithOptions { get; set; } = [];
    public ISqlExpression? AsSelect { get; set; }

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        if (AsSelect != null)
        {
            sql.Write($"CREATE TABLE {TableName} AS {AsSelect.ToSql()}");
            return sql.ToString();
        }

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
        if (!string.IsNullOrEmpty(Period))
        {
            sql.WriteLine($"PERIOD FOR SYSTEM_TIME ({Period})");
        }

        sql.Indent--;
        sql.WriteLine(")");
        if (!string.IsNullOrEmpty(OnFileGroup))
        {
            sql.WriteLine($"ON {OnFileGroup}");
        }

        if (!string.IsNullOrEmpty(TextImageOn))
        {
            sql.WriteLine($"TEXTIMAGE_ON {TextImageOn}");
        }

        if (WithOptions.Count > 0)
        {
            sql.WriteLine($"WITH ({string.Join(", ", WithOptions)})");
        }

        return sql.ToString();
    }
}