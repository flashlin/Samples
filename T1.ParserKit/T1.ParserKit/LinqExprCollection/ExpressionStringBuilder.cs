using System.Linq.Expressions;

namespace T1.ParserKit.LinqExprCollection;

public class ExpressionStringBuilder : ExpressionVisitor
{
    private readonly Stack<string> _expressionStack = new Stack<string>();

    public string ToString(Expression expression)
    {
        Visit(expression);
        return _expressionStack.Pop();
    }

    protected override Expression VisitMethodCall(MethodCallExpression node)
    {
        if (node.Method.Name == "Where" && node.Method.DeclaringType == typeof(Queryable))
        {
            var lambda = (LambdaExpression)node.Arguments[1];
            Visit(lambda.Body);
            _expressionStack.Push($"WHERE {_expressionStack.Pop()}");
            return node;
        }

        return base.VisitMethodCall(node);
    }

    protected override Expression VisitBinary(BinaryExpression node)
    {
        var leftOperand = VisitOperand(node.Left);
        var rightOperand = VisitOperand(node.Right);
        var operatorSymbol = GetOperatorSymbol(node.NodeType);
        _expressionStack.Push($"({leftOperand} {operatorSymbol} {rightOperand})");
        return node;
    }

    private string VisitOperand(Expression operand)
    {
        if (operand is ConstantExpression constantExpression)
        {
            return constantExpression.Value.ToString();
        }
        else if (operand is MemberExpression memberExpression)
        {
            return memberExpression.Member.Name;
        }

        Visit(operand);
        return _expressionStack.Pop();
    }

    private string GetOperatorSymbol(ExpressionType nodeType)
    {
        switch (nodeType)
        {
            case ExpressionType.Equal:
                return "==";
            case ExpressionType.NotEqual:
                return "!=";
            case ExpressionType.GreaterThan:
                return ">";
            case ExpressionType.GreaterThanOrEqual:
                return ">=";
            case ExpressionType.LessThan:
                return "<";
            case ExpressionType.LessThanOrEqual:
                return "<=";
            case ExpressionType.AndAlso:
                return "&&";
            case ExpressionType.OrElse:
                return "||";
            default:
                return nodeType.ToString();
        }
    }
}