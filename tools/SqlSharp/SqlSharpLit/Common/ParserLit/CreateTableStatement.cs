using System.Text;
using Microsoft.Extensions.Primitives;

namespace SqlSharpLit.Common.ParserLit;

public enum ReferentialAction
{
    NoAction,
    Cascade,
    SetNull,
    SetDefault
}

public static class ReferentialActionExtensions
{
    public static string ToSql(this ReferentialAction action)
    {
        return action switch
        {
            ReferentialAction.Cascade => "CASCADE",
            ReferentialAction.SetNull => "SET NULL",
            ReferentialAction.SetDefault => "SET DEFAULT",
            _ => "NO ACTION"
        };
    }
}

public class SqlTableForeignKeyExpression : ISqlExpression
{
    public string ConstraintName { get; set; } = string.Empty;
    public SqlType SqlType { get; }= SqlType.TableForeignKey;
    public List<SqlConstraintColumn> Columns { get; set; } = [];
    public string ReferencedTableName { get; set; } = string.Empty;
    public string RefColumn { get; set; } = string.Empty; 
    public ReferentialAction OnDeleteAction { get; set; } = ReferentialAction.NoAction;
    public ReferentialAction OnUpdateAction { get; set; } = ReferentialAction.NoAction;
    public bool NotForReplication { get; set; }

    public string ToSql()
    {
        var sql = new StringBuilder();
        if (!string.IsNullOrEmpty(ConstraintName))
        {
            sql.Append($"CONSTRAINT {ConstraintName} ");
        }

        sql.Append("(");
        sql.Append(string.Join(", ", Columns.Select(c => c.ToSql())));
        sql.Append(")");
        
        sql.Append($" REFERENCES {ReferencedTableName}");
        if (!string.IsNullOrEmpty(RefColumn))
        {
            sql.Append($"({RefColumn})");
        }
        if (OnDeleteAction != ReferentialAction.NoAction)
        {
            sql.Append($" ON DELETE {OnDeleteAction.ToSql()}");
        }
        if (OnUpdateAction != ReferentialAction.NoAction)
        {
            sql.Append($" ON UPDATE {OnUpdateAction.ToSql()}");
        }
        if(NotForReplication)
        {
            sql.Append(" NOT FOR REPLICATION");
        }
        return sql.ToString();
    }
}

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