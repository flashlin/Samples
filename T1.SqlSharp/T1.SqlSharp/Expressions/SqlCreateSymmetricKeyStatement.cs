using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlCreateSymmetricKeyStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateSymmetricKeyStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateSymmetricKeyStatement(this);
    }

    public string Name { get; set; } = string.Empty;
    public List<string> Options { get; set; } = [];

    public string ToSql()
    {
        var options = Options.Count > 0 ? $" WITH {string.Join(", ", Options)}" : string.Empty;
        return $"CREATE SYMMETRIC KEY {Name}{options}";
    }
}
