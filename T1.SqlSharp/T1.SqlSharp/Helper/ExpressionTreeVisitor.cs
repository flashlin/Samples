using System.Linq.Expressions;
using T1.SqlSharp.Expressions;

namespace T1.SqlSharp.Helper;

public class ExpressionTreeVisitor
{
    private readonly SqlExpressionBuilderContext _context;
    private int _parameterIndex = 0;

    public ExpressionTreeVisitor(SqlExpressionBuilderContext context)
    {
        _context = context;
    }

    public ISqlExpression Visit(Expression expression)
    {
        return expression.NodeType switch
        {
            ExpressionType.Equal => VisitBinary((BinaryExpression)expression),
            ExpressionType.NotEqual => VisitBinary((BinaryExpression)expression),
            ExpressionType.GreaterThan => VisitBinary((BinaryExpression)expression),
            ExpressionType.GreaterThanOrEqual => VisitBinary((BinaryExpression)expression),
            ExpressionType.LessThan => VisitBinary((BinaryExpression)expression),
            ExpressionType.LessThanOrEqual => VisitBinary((BinaryExpression)expression),
            ExpressionType.MemberAccess => VisitMember((MemberExpression)expression),
            ExpressionType.Constant => VisitConstant((ConstantExpression)expression),
            _ => throw new NotSupportedException($"不支援的表達式類型：{expression.NodeType}")
        };
    }

    private ISqlExpression VisitBinary(BinaryExpression node)
    {
        var left = Visit(node.Left);
        var right = Visit(node.Right);
        var comparisonOperator = MapExpressionType(node.NodeType);

        return new SqlConditionExpression
        {
            Left = left,
            ComparisonOperator = comparisonOperator,
            Right = right
        };
    }

    private ISqlExpression VisitMember(MemberExpression node)
    {
        if (node.Expression is ParameterExpression)
        {
            return new SqlColumnExpression
            {
                Schema = _context.Schema,
                TableName = _context.TableName,
                ColumnName = node.Member.Name
            };
        }

        var value = GetMemberValue(node);
        return CreateParameter(value);
    }

    private ISqlExpression VisitConstant(ConstantExpression node)
    {
        return CreateParameter(node.Value);
    }

    private object? GetMemberValue(MemberExpression node)
    {
        var objectMember = Expression.Convert(node, typeof(object));
        var getterLambda = Expression.Lambda<Func<object>>(objectMember);
        var getter = getterLambda.Compile();
        return getter();
    }

    private ISqlExpression CreateParameter(object? value)
    {
        var paramName = $"@p{_parameterIndex++}";
        _context.Parameters[paramName] = value;

        return new SqlParameter
        {
            ParameterName = paramName,
            Value = value
        };
    }

    private ComparisonOperator MapExpressionType(ExpressionType type)
    {
        return type switch
        {
            ExpressionType.Equal => ComparisonOperator.Equal,
            ExpressionType.NotEqual => ComparisonOperator.NotEqual,
            ExpressionType.GreaterThan => ComparisonOperator.GreaterThan,
            ExpressionType.LessThan => ComparisonOperator.LessThan,
            ExpressionType.GreaterThanOrEqual => ComparisonOperator.GreaterThanOrEqual,
            ExpressionType.LessThanOrEqual => ComparisonOperator.LessThanOrEqual,
            _ => throw new NotSupportedException()
        };
    }
}
