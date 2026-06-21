using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseCreateViewSqlTest
{
    [Test]
    public void Create_view_simple()
    {
        var sql = "CREATE VIEW vCustomer AS SELECT Id, Name FROM Customer";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateViewStatement
        {
            ViewName = "vCustomer",
            Query = new SelectStatement
            {
                Columns =
                [
                    new SelectColumn { Field = new SqlFieldExpr { FieldName = "Id" } },
                    new SelectColumn { Field = new SqlFieldExpr { FieldName = "Name" } }
                ],
                FromSources = [new SqlTableSource { TableName = "Customer" }]
            }
        });
    }

    [Test]
    public void Create_or_alter_view()
    {
        var sql = "CREATE OR ALTER VIEW vCustomer AS SELECT Id FROM Customer";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateViewStatement
        {
            IsOrAlter = true,
            ViewName = "vCustomer",
            Query = new SelectStatement
            {
                Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "Id" } }],
                FromSources = [new SqlTableSource { TableName = "Customer" }]
            }
        });
    }

    [Test]
    public void Create_view_with_column_list()
    {
        var sql = "CREATE VIEW vCustomer (a, b) AS SELECT Id, Name FROM Customer";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateViewStatement
        {
            ViewName = "vCustomer",
            ColumnNames = ["a", "b"],
            Query = new SelectStatement
            {
                Columns =
                [
                    new SelectColumn { Field = new SqlFieldExpr { FieldName = "Id" } },
                    new SelectColumn { Field = new SqlFieldExpr { FieldName = "Name" } }
                ],
                FromSources = [new SqlTableSource { TableName = "Customer" }]
            }
        });
    }

    [Test]
    public void Create_view_with_check_option()
    {
        var sql = "CREATE VIEW vCustomer AS SELECT Id FROM Customer WHERE Id > 0 WITH CHECK OPTION";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateViewStatement
        {
            ViewName = "vCustomer",
            WithCheckOption = true,
            Query = new SelectStatement
            {
                Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "Id" } }],
                FromSources = [new SqlTableSource { TableName = "Customer" }],
                Where = new SqlConditionExpression
                {
                    Left = new SqlFieldExpr { FieldName = "Id" },
                    ComparisonOperator = ComparisonOperator.GreaterThan,
                    Right = new SqlValue { SqlType = SqlType.IntValue, Value = "0" }
                }
            }
        });
    }
}
