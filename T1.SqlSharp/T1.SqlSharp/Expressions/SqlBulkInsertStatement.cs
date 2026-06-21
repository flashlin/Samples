using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlBulkInsertStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.BulkInsertStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_BulkInsertStatement(this);
    }

    public string TableName { get; set; } = string.Empty;
    public string DataFile { get; set; } = string.Empty;
    public List<string> Options { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"BULK INSERT {TableName} FROM {DataFile}");
        if (Options.Count > 0)
        {
            sql.Append($" WITH ({string.Join(", ", Options)})");
        }

        return sql.ToString();
    }
}
