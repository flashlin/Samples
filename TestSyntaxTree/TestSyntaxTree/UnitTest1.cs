using System.Text;
using FluentAssertions;

namespace TestSyntaxTree;

// https://itnext.io/syntax-tree-and-alternative-to-linq-in-interaction-with-sql-databases-656b78fe00dc

public class Tests
{
    [SetUp]
    public void Setup()
    {
    }

    [Test]
    public void Test1()
    {
        var firstName = new ExprColumn{Name = "FirstName"};
        var lastName = new ExprColumn{Name = "LastName"};

        var expr = firstName == "John" & (lastName == "Smith" | lastName == "Doe");
        var sqlBuilder = new MySqlBuilder();
        expr.Accept(sqlBuilder);
        var sql = sqlBuilder.GetResult();
        sql.Should().Be("[FirstName]='John' AND ([LastName]='Smith' OR [LastName]='Doe')");
    }
}

public abstract class Expr
{
    public abstract void Accept(IExprVisitor visitor);
}

public class ExprColumn : Expr
{
    public string Name { get; set; }
    public override void Accept(IExprVisitor visitor)
    {
        visitor.VisitColumn(this);
    }
    
    public static ExprBoolean operator==(ExprColumn column, string value)
        => new ExprEqPredicate 
        {
            Column = column, 
            Value = new ExprStr
            {
                Value = value
            }
        };

    public static ExprBoolean operator !=(ExprColumn column, string value)
        => new ExprNotEqPredicate
        {
            Column = column, 
            Value = new ExprStr
            {
                Value = value
            }
        };
}

public class ExprStr : Expr
{
    public string Value { get; set; }
    public override void Accept(IExprVisitor visitor)
    {
        visitor.VisitStr(this);
    }
}

public abstract class ExprBoolean : Expr
{
    public static ExprBoolean operator &(ExprBoolean left, ExprBoolean right)
        => new ExprAnd
        {
            Left = left, 
            Right = right
        };

    public static ExprBoolean operator |(ExprBoolean left, ExprBoolean right)
        => new ExprOr
        {
            Left = left, 
            Right = right
        };
}

public class ExprEqPredicate : ExprBoolean
{
    public ExprColumn Column { get; set; }
    public Expr Value { get; set; }
    public override void Accept(IExprVisitor visitor)
    {
        visitor.VisitEqPredicate(this);
    }
}

public class ExprNotEqPredicate : ExprBoolean
{
    public ExprColumn Column { get; set; }
    public Expr Value { get; set; }
    public override void Accept(IExprVisitor visitor)
    {
        visitor.VisitNotEqPredicate(this);
    }
}

public class ExprAnd : ExprBoolean
{
    public ExprBoolean Left { get; set; }
    public ExprBoolean Right { get; set; }
    public override void Accept(IExprVisitor visitor)
    {
        visitor.VisitAnd(this);
    }
}

public class ExprOr : ExprBoolean
{
    public ExprBoolean Left { get; set; }
    public ExprBoolean Right { get; set; }
    public override void Accept(IExprVisitor visitor)
    {
        visitor.VisitOr(this);
    }
}

public interface IExprVisitor
{
    void VisitColumn(ExprColumn column);
    void VisitStr(ExprStr str);
    void VisitEqPredicate(ExprEqPredicate eqPredicate);
    void VisitOr(ExprOr expr);
    void VisitAnd(ExprAnd and);
    void VisitNotEqPredicate(ExprNotEqPredicate exprNotEqPredicate);
}

public class MySqlBuilder : IExprVisitor
{
    private readonly StringBuilder _stringBuilder 
        = new StringBuilder();

    public string GetResult()
        => this._stringBuilder.ToString();

    public void VisitColumn(ExprColumn column)
    {
        _stringBuilder.Append('[' + Escape(column.Name, ']') + ']');
    }

    public void VisitStr(ExprStr str)
    {
        _stringBuilder.Append('\'' + Escape(str.Value, '\'') + '\'');
    }

    public void VisitEqPredicate(ExprEqPredicate eqPredicate)
    {
        eqPredicate.Column.Accept(this);
        _stringBuilder.Append('=');
        eqPredicate.Value.Accept(this);
    }

    public void VisitAnd(ExprAnd and)
    {
        and.Left.Accept(this);
        _stringBuilder.Append(" AND ");
        and.Right.Accept(this);
    }

    public void VisitNotEqPredicate(ExprNotEqPredicate expr)
    {
        expr.Column.Accept(this);
        _stringBuilder.Append(" != ");
        expr.Value.Accept(this);
    }

    public void VisitOr(ExprOr expr)
    {
        ParenthesisForOr(expr.Left);
        _stringBuilder.Append(" OR ");
        ParenthesisForOr(expr.Right);
    }

    void ParenthesisForOr(ExprBoolean expr)
    {
        if (expr is ExprOr)
        {
            _stringBuilder.Append('(');
            expr.Accept(this);
            _stringBuilder.Append(')');
        }
        else
        {
            expr.Accept(this);
        }
    }

    private static string Escape(string str, char ch)
    {
        return str;
    }
}