using System.Linq.Expressions;
using System.Text;

namespace T1.LinqSqlBuildEx;

public class LinqSqlBuilder
{
}

public interface ILinqSqlExpr
{
    string Build();
}

public class From<TEntity> : ILinqSqlExpr
{
    private string _tableName = typeof(TEntity).Name;
    private List<ColumnInfo> _columns = new();

    public string Build()
    {
        var sb = new StringBuilder();
        sb.Append("SELECT ");
        foreach (var (column,idx) in _columns.Select((val,idx)=> (val,idx)))
        {
            if (idx != 0)
            {
                sb.Append(",");
            }
            sb.Append($"tb1.{column.Name} as {column.Alias}");
        }

        sb.Append($" FROM {_tableName} as tb1 WITH(NOLOCK)");
        return sb.ToString();
    }

    public From<TEntity> Select<TResult>(Expression<Func<TEntity, TResult>> columnSelector)
    {
        _columns = columnSelector.ExtractColumns().ToList();
        return this;
    }
}

public class ColumnInfo
{
    public string Name { get; set; }
    public string Alias { get; set; }
}

public static class Extractor
{
    public static IEnumerable<ColumnInfo> ExtractColumns<TEntity, TResult>(this Expression<Func<TEntity, TResult>> columnSelector)
    {
        if (columnSelector.Body is MemberExpression memberExpression)
        {
            // Case 1 and 2: x => x or x => x.Id
            yield return new ColumnInfo
            {
                Name = memberExpression.Member.Name,
                Alias = memberExpression.Member.Name
            };
            yield break;
        }

        if (columnSelector.Body is NewExpression newExpression)
        {
            foreach (var argument in newExpression.Arguments)
            {
                if (argument is MemberExpression memberExpr)
                {
                    var propertyName = memberExpr.Member.Name;
                    var aliasName = propertyName;
                    yield return new ColumnInfo()
                    {
                        Name = propertyName,
                        Alias = aliasName
                    };
                }

                if (argument is {NodeType: ExpressionType.MemberInit})
                {
                    var memberInitExpr = (MemberInitExpression) argument;
                    var memberAssignments = memberInitExpr.Bindings
                        .OfType<MemberAssignment>()
                        .Select(binding => new ColumnInfo()
                        {
                            Name = binding.Member.Name,
                            Alias = binding.Member.Name
                        });
                    foreach (var memberAssignment in memberAssignments)
                    {
                        yield return memberAssignment;
                    }
                }
            }
        }
    }
}

public static class Sql
{
    public static From<TEntity> From<TEntity>()
    {
        return new From<TEntity>();
    }
}