using System.Linq.Expressions;
using System.Text;

namespace SqlBoyLib;

public class SqlQueryBuilder<TEntity> where TEntity : class
{
    private readonly string _tableName;
    private readonly List<string> _selectColumns = new();
    private readonly List<string> _whereConditions = new();
    private readonly List<string> _orderByColumns = new();
    private readonly Dictionary<string, SqlParameter> _parameters = new();
    private int _parameterCounter = 0;

    public SqlQueryBuilder(string tableName)
    {
        _tableName = tableName;
    }

    public SqlQueryBuilder<TEntity> Where(Expression<Func<TEntity, bool>> predicate)
    {
        var whereClause = ParseExpression(predicate.Body);
        _whereConditions.Add(whereClause);
        return this;
    }

    public SqlQueryBuilder<TEntity> OrderBy(Expression<Func<TEntity, object>> keySelector)
    {
        var columnName = GetMemberName(keySelector.Body);
        _orderByColumns.Add($"{columnName} ASC");
        return this;
    }

    public SqlQueryBuilder<TEntity> OrderByDescending(Expression<Func<TEntity, object>> keySelector)
    {
        var columnName = GetMemberName(keySelector.Body);
        _orderByColumns.Add($"{columnName} DESC");
        return this;
    }

    public SqlQuery Build()
    {
        var sb = new StringBuilder();
        sb.Append("SELECT * FROM ");
        sb.Append(_tableName);

        if (_whereConditions.Any())
        {
            sb.Append(" WHERE ");
            sb.Append(string.Join(" AND ", _whereConditions));
        }

        if (_orderByColumns.Any())
        {
            sb.Append(" ORDER BY ");
            sb.Append(string.Join(", ", _orderByColumns));
        }

        var paramDefinitions = string.Join(", ", _parameters.Select(p => $"{p.Key} {p.Value.SqlType}"));
        var paramValues = _parameters.ToDictionary(p => p.Key, p => p.Value.Value);

        return new SqlQuery
        {
            Statement = sb.ToString(),
            ParameterDefinitions = paramDefinitions,
            Parameters = paramValues
        };
    }

    private string ParseExpression(Expression expression)
    {
        switch (expression)
        {
            case BinaryExpression binary:
                return ParseBinaryExpression(binary);
            
            case MemberExpression member:
                return GetMemberName(member);
            
            case ConstantExpression constant:
                return AddParameter(constant.Value, constant.Type);
            
            case UnaryExpression unary when unary.NodeType == ExpressionType.Not:
                return $"NOT ({ParseExpression(unary.Operand)})";
            
            case UnaryExpression unary when unary.NodeType == ExpressionType.Convert:
                return ParseExpression(unary.Operand);
            
            default:
                throw new NotSupportedException($"Expression type {expression.NodeType} is not supported");
        }
    }

    private string ParseBinaryExpression(BinaryExpression binary)
    {
        var left = ParseExpression(binary.Left);
        var right = ParseExpression(binary.Right);

        return binary.NodeType switch
        {
            ExpressionType.Equal => $"{left} = {right}",
            ExpressionType.NotEqual => $"{left} <> {right}",
            ExpressionType.GreaterThan => $"{left} > {right}",
            ExpressionType.GreaterThanOrEqual => $"{left} >= {right}",
            ExpressionType.LessThan => $"{left} < {right}",
            ExpressionType.LessThanOrEqual => $"{left} <= {right}",
            ExpressionType.AndAlso => $"({left} AND {right})",
            ExpressionType.OrElse => $"({left} OR {right})",
            _ => throw new NotSupportedException($"Binary operator {binary.NodeType} is not supported")
        };
    }

    private string GetMemberName(Expression expression)
    {
        switch (expression)
        {
            case MemberExpression member:
                return member.Member.Name;
            
            case UnaryExpression unary when unary.NodeType == ExpressionType.Convert:
                return GetMemberName(unary.Operand);
            
            default:
                throw new NotSupportedException($"Cannot extract member name from expression type {expression.NodeType}");
        }
    }

    private string AddParameter(object? value, Type type)
    {
        _parameterCounter++;
        var paramName = $"@p{_parameterCounter}";
        
        _parameters[paramName] = new SqlParameter
        {
            Name = paramName,
            SqlType = SqlParameter.GetSqlType(type),
            Value = value
        };

        return paramName;
    }
}

