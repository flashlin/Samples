using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlCreateProcedureStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateProcedureStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateProcedureStatement(this);
    }

    public bool IsOrAlter { get; set; }
    public bool IsAlter { get; set; }
    public string ProcedureName { get; set; } = string.Empty;
    public List<SqlProcedureParameter> Parameters { get; set; } = [];
    public List<string> Options { get; set; } = [];
    public required ISqlExpression Body { get; set; }

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append(DefinitionLead.ToSql(IsAlter, IsOrAlter, "PROCEDURE"));
        sql.Append(ProcedureName);
        if (Parameters.Count > 0)
        {
            sql.Append($" {string.Join(", ", Parameters.Select(FormatParameter))}");
        }
        if (Options.Count > 0)
        {
            sql.Append($" WITH {string.Join(", ", Options)}");
        }
        sql.Append($" AS {Body.ToSql()}");
        return sql.ToString();
    }

    private static string FormatParameter(SqlProcedureParameter parameter)
    {
        var text = $"{parameter.Name} {parameter.DataType}";
        if (parameter.DataSize != null)
        {
            text += parameter.DataSize.ToSql();
        }
        if (parameter.DefaultValue != null)
        {
            text += $" = {parameter.DefaultValue.ToSql()}";
        }
        if (parameter.IsOutput)
        {
            text += " OUTPUT";
        }
        if (parameter.IsReadOnly)
        {
            text += " READONLY";
        }
        return text;
    }
}
