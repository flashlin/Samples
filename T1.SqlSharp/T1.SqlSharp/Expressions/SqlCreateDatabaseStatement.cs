using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlCreateDatabaseStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateDatabaseStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateDatabaseStatement(this);
    }

    public string DatabaseName { get; set; } = string.Empty;
    public string Containment { get; set; } = string.Empty;
    public bool OnPrimary { get; set; }
    public List<string> DataFiles { get; set; } = [];
    public List<SqlDatabaseFileGroup> FileGroups { get; set; } = [];
    public List<string> LogFiles { get; set; } = [];
    public string Collation { get; set; } = string.Empty;
    public List<string> Options { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"CREATE DATABASE {DatabaseName}");
        if (!string.IsNullOrEmpty(Containment))
        {
            sql.Append($" CONTAINMENT = {Containment}");
        }

        if (DataFiles.Count > 0)
        {
            sql.Append(OnPrimary ? " ON PRIMARY " : " ON ");
            sql.Append(string.Join(", ", DataFiles));
        }

        foreach (var fileGroup in FileGroups)
        {
            sql.Append($", {fileGroup.ToSql()}");
        }

        if (LogFiles.Count > 0)
        {
            sql.Append($" LOG ON {string.Join(", ", LogFiles)}");
        }

        if (!string.IsNullOrEmpty(Collation))
        {
            sql.Append($" COLLATE {Collation}");
        }

        if (Options.Count > 0)
        {
            sql.Append($" WITH {string.Join(", ", Options)}");
        }

        return sql.ToString();
    }
}
