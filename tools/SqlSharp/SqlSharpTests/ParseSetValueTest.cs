using T1.SqlSharp.Expressions;

namespace SqlSharpTests;

[TestFixture]
public class ParseSetValueTest
{
    [Test]
    public void Set_String()
    {
        var sql = $"""
                   set @name = 'test'
                   """;
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlSetValueStatement()
        {
            Name = new SqlFieldExpr { FieldName = "@name" },
            Value = new SqlValue { Value = "'test'" }
        });
    }
}