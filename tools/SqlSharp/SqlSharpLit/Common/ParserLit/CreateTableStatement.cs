namespace SqlSharpLit.Common.ParserLit;

public class CreateTableStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateTable;
    public string TableName { get; set; } = string.Empty;
    public List<ColumnDefinition> Columns { get; set; } = [];
    public List<ISqlExpression> Constraints { get; set; } = [];

    public string ToSql()
    {
        var sql = new T1.Standard.IO.IndentStringBuilder();
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