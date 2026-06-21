using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlDeclareStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.DeclareStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_DeclareStatement(this);
    }

    public List<SqlVariableDeclaration> Declarations { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append("DECLARE ");
        sql.Append(string.Join(", ", Declarations.Select(FormatDeclaration)));
        return sql.ToString();
    }

    private static string FormatDeclaration(SqlVariableDeclaration declaration)
    {
        var text = $"{declaration.Name} {declaration.DataType}";
        if (declaration.DataSize != null)
        {
            text += declaration.DataSize.ToSql();
        }
        if (declaration.InitialValue != null)
        {
            text += $" = {declaration.InitialValue.ToSql()}";
        }
        return text;
    }
}
