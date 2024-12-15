using System.Text;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharp.Expressions;

public class SqlConstraintForeignKey : ISqlConstraint
{
    public SqlType SqlType { get; }= SqlType.TableForeignKey;
    public TextSpan Span { get; set; } = new();
    public string ConstraintName { get; set; } = string.Empty;
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