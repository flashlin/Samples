using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseValuesTableSourceSqlTest
{
    [Test]
    public void Select_from_values_constructor()
    {
        var sql = "SELECT c1 FROM (VALUES (1, 'a'), (2, 'b')) AS t (c1, c2)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "c1" } }],
            FromSources =
            [
                new SqlValuesTableSource
                {
                    Alias = "t",
                    ColumnAliases = ["c1", "c2"],
                    Rows =
                    [
                        [
                            new SqlValue { SqlType = SqlType.IntValue, Value = "1" },
                            new SqlValue { SqlType = SqlType.String, Value = "'a'" }
                        ],
                        [
                            new SqlValue { SqlType = SqlType.IntValue, Value = "2" },
                            new SqlValue { SqlType = SqlType.String, Value = "'b'" }
                        ]
                    ]
                }
            ]
        });
    }
}
