using SqlSharpLit.Common.ParserLit;
using T1.SqlSharp.Expressions;
using T1.Standard.DesignPatterns;

namespace SqlSharpTests;

[TestFixture]
public class ParseSelectSqlTest
{
    [Test]
    public void WithNoLock()
    {
        var sql = $"""
                   select id from customer with(nolock)
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SelectStatement
        {
            Columns = [
                new SelectColumn
                {
                    ColumnName = "id",
                }
            ],
            From = new SqlTableSource
            {
                TableName = "customer",
                Withs = [
                    new SqlHint()
                    {
                        Name = "nolock",
                    }
                ],
            },
        });
    }
    
    [Test]
    public void METHOD()
    {
        var sql = $"""
                   select top (@batchsize) transid from bettrans with(nolock) where matchid = @matchid and bettype < 39 and WinLostDate <> @eventDate
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SelectStatement
        {
            Top = new SqlTopClause
            {
                Expression = new SqlValue(),
            },
            Columns = [
                new SelectColumn
                {
                    ColumnName = "transid",
                }
            ],
            From = new SqlTableSource
            {
                TableName = "bettrans",
                Withs = [
                    new SqlHint()
                    {
                        Name = "nolock",
                    }
                ],
            },
            Where = new SqlSearchCondition
            {
                Left = new SqlConditionExpression
                {
                    Left = null,
                    ComparisonOperator = ComparisonOperator.Equal,
                    Right = null
                },
                LogicalOperator = LogicalOperator.None,
                Right = null
            }
        });
}
    
    [Test]
    public void Where_FunctionName()
    {
        var sql = $"""
                   select 1
                   from sys.databases
                   where name = DB_NAME()
                   and SUSER_SNAME(owner_sid) = 'sa'
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn { ColumnName = "1" },
            ],
            From = new SqlTableSource
            {
                TableName = "sys.databases",
            },
            Where = new SqlSearchCondition()
            {
                Left = new SqlConditionExpression
                {
                    Left = new SqlFieldExpression
                    {
                        FieldName = "name",
                    },
                    ComparisonOperator = ComparisonOperator.Equal,
                    Right = new SqlFunctionExpression
                    {
                        FunctionName = "DB_NAME",
                    }
                },
                LogicalOperator = LogicalOperator.And,
                Right = new SqlConditionExpression
                {
                    Left = new SqlFunctionExpression
                    {
                        FunctionName = "SUSER_SNAME",
                        Parameters =
                        [
                            new SqlFieldExpression
                            {
                                FieldName = "owner_sid",
                            }
                        ]
                    },
                    ComparisonOperator = ComparisonOperator.Equal,
                    Right = new SqlValue
                    {
                        Value = "'sa'"
                    }
                }
            }
        });
    }

    [Test]
    public void Select()
    {
        var sql = $"""
                   SELECT Id, Name 
                   FROM Persons
                   WHERE Id = 1;
                   """;

        var rc = ParseSql(sql);

        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn { ColumnName = "Id" },
                new SelectColumn { ColumnName = "Name" },
            ],
            From = new SqlTableSource
            {
                TableName = "Persons"
            },
            Where = new SqlConditionExpression
            {
                Left = new SqlFieldExpression
                {
                    FieldName = "Id",
                },
                ComparisonOperator = ComparisonOperator.Equal,
                Right = new SqlValue
                {
                    SqlType = SqlType.IntValue,
                    Value = "1"
                }
            }
        });
    }

    private static ParseResult<ISqlExpression> ParseSql(string sql)
    {
        var p = new SqlParser(sql);
        return p.Parse();
    }
}