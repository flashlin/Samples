using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseCteDmlTest
{
    [Test]
    public void Cte_prefix_insert()
    {
        var sql = "WITH cte AS (SELECT id FROM src) INSERT INTO dst (id) SELECT id FROM cte";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlWithCte
        {
            CommonTableExpressions =
            [
                new SqlCommonTableExpression
                {
                    Name = "cte",
                    Query = new SelectStatement
                    {
                        Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }],
                        FromSources = [new SqlTableSource { TableName = "src" }]
                    }
                }
            ],
            Statement = new SqlInsertStatement
            {
                TableName = "dst",
                Columns = ["id"],
                SourceSelect = new SelectStatement
                {
                    Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }],
                    FromSources = [new SqlTableSource { TableName = "cte" }]
                }
            }
        });
    }

    [Test]
    public void Cte_prefix_update()
    {
        var sql = "WITH cte AS (SELECT id FROM src) UPDATE dst SET Name = 'x'";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlWithCte
        {
            CommonTableExpressions =
            [
                new SqlCommonTableExpression
                {
                    Name = "cte",
                    Query = new SelectStatement
                    {
                        Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }],
                        FromSources = [new SqlTableSource { TableName = "src" }]
                    }
                }
            ],
            Statement = new SqlUpdateStatement
            {
                TableName = "dst",
                SetClauses =
                [
                    new SqlAssignExpr
                    {
                        Left = new SqlFieldExpr { FieldName = "Name" },
                        Right = new SqlValue { SqlType = SqlType.String, Value = "'x'" }
                    }
                ]
            }
        });
    }

    [Test]
    public void Cte_prefix_delete()
    {
        var sql = "WITH cte AS (SELECT id FROM src) DELETE FROM dst WHERE id = 1";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlWithCte
        {
            CommonTableExpressions =
            [
                new SqlCommonTableExpression
                {
                    Name = "cte",
                    Query = new SelectStatement
                    {
                        Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }],
                        FromSources = [new SqlTableSource { TableName = "src" }]
                    }
                }
            ],
            Statement = new SqlDeleteStatement
            {
                TableName = "dst",
                Where = new SqlConditionExpression
                {
                    Left = new SqlFieldExpr { FieldName = "id" },
                    ComparisonOperator = ComparisonOperator.Equal,
                    Right = new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
                }
            }
        });
    }
}
