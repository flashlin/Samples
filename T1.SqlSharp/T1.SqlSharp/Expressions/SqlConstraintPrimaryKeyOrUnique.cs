using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlConstraintPrimaryKeyOrUnique : ISqlConstraint
{
    public SqlType SqlType => SqlType.Constraint;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_ConstraintPrimaryKeyOrUnique(this);
    }

    public string ConstraintName { get; set; } = string.Empty;
    public string ConstraintType { get; set; } = string.Empty;
    public string Clustered { get; set; } = string.Empty;
    public List<SqlConstraintColumn> Columns { get; set; } = [];
    public List<SqlToggle> WithToggles { get; set; } = [];
    public string On { get; set; } = string.Empty;
    public SqlIdentity Identity { get; set; } = SqlIdentity.Default;

    public string ToSql()
    {
        var sb = new IndentStringBuilder();
        if (!string.IsNullOrEmpty(ConstraintName))
        {
            sb.Write($"CONSTRAINT {ConstraintName} {ConstraintType}");
        }
        else
        {
            sb.Write($"{ConstraintType}");
        }
        if (!string.IsNullOrEmpty(Clustered))
        {
            sb.Write($" {Clustered}");
        }
        if (Columns.Count > 0)
        {
            sb.Write(" (");
            sb.Write(string.Join(", ", Columns.Select(c => c.ToSql())));
            sb.Write(")");
        }
        if (WithToggles.Count > 0)
        {
            sb.Write(" WITH (");
            sb.Write(string.Join(", ", WithToggles.Select(t => t.ToSql())));
            sb.Write(")");
        }
        if(Identity != SqlIdentity.Default)
        {
            sb.Write($" {Identity.ToSql()}");
        }
        if (!string.IsNullOrEmpty(On))
        {
            sb.Write($" ON {On}");
        }
        return sb.ToString();
    }
}