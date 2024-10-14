using FluentAssertions;
using T1.ParserKit.Core;
using T1.ParserKit.LinqExprCollection;

namespace ParserKitTests;

public class LinqExpressionTest
{
    [Test]
    public void get_linq_expr_string()
    {
        var customers = new List<User>();
        var q1 = from tb1 in customers where tb1.Age > 10 select tb1;
        var queryable = q1.AsQueryable().Expression;
        var expr = new ExpressionStringBuilder().ParseToString(queryable);
        expr.Should().Be("from tb1 in customers select tb1.Id");
    }
    
    [Test]
    public void from_tb1_()
    {
        //var q1 = "from tb1 in customers where tb1.Age > 10 select tb1";
        var q1 = "from tb1";
        var parser = new LinqGrammar();
        var expr = parser.Parse(q1);
        expr.Should().Be(new LinqSelectExpr
        {
            AliasTableName = "tb1"
        });
    }

    public class User
    {
        public int Id { get; set; }
        public string Name { get; set; } = string.Empty;
        public int Age { get; set; }
    }
}