using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlCreatePartitionSchemeStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreatePartitionSchemeStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreatePartitionSchemeStatement(this);
    }

    public string SchemeName { get; set; } = string.Empty;
    public string PartitionFunction { get; set; } = string.Empty;
    public bool AllToOneFileGroup { get; set; }
    public List<string> FileGroups { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"CREATE PARTITION SCHEME {SchemeName} AS PARTITION {PartitionFunction}");
        if (AllToOneFileGroup)
        {
            sql.Append(" ALL");
        }

        sql.Append($" TO ({string.Join(", ", FileGroups)})");
        return sql.ToString();
    }
}
