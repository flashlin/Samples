using T1.SqlSharp.Expressions;

namespace SqlSharpTests;

[TestFixture]
public class ExcludeNonSelectStatementTest
{
    [Test]
    public void ExtractKnownStatements()
    {
        var sql = $"""
                   print '123'
                   set name='123'
                   select 1
                   """;
        var rc = sql.ExtractStatements().ToList();
        rc.ShouldBeList([
            new SqlSetValueStatement()
            {
                Name = new SqlFieldExpr { FieldName = "name" },
                Value = new SqlValue { Value = "'123'" }
            },
            new SelectStatement
            {
                Columns =
                [
                    new SelectColumn()
                    {
                        Field = new SqlValue
                        {
                            SqlType = SqlType.IntValue,
                            Value = "1"
                        }
                    }
                ]
            }
        ]);
    }
}