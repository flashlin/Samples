using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlCreateCertificateStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateCertificateStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateCertificateStatement(this);
    }

    public string Name { get; set; } = string.Empty;
    public string FromFile { get; set; } = string.Empty;
    public string Password { get; set; } = string.Empty;
    public List<string> Options { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"CREATE CERTIFICATE {Name}");
        if (!string.IsNullOrEmpty(FromFile))
        {
            sql.Append($" FROM FILE = {FromFile}");
        }

        if (!string.IsNullOrEmpty(Password))
        {
            sql.Append($" ENCRYPTION BY PASSWORD = {Password}");
        }

        if (Options.Count > 0)
        {
            sql.Append($" WITH {string.Join(", ", Options)}");
        }

        return sql.ToString();
    }
}
