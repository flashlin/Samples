using SqlSharpLit.Common.ParserLit;
using T1.SqlSharp.Expressions;
using T1.Standard.DesignPatterns;

namespace SqlSharpTests;

[TestFixture]
public class ParseSelectSqlTest
{
    [Test]
    public void Order_By_DESC()
    {
        var sql = $"""
                   select top 1 UserName 
                   from customer 
                   where UserName=@username 
                   order by LastDate desc
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SelectStatement
        {
            Top = new SqlTopClause
            {
                Expression = new SqlValue
                {
                    SqlType = SqlType.IntValue,
                    Value = "1"
                }
            },
            Columns = [
                new SelectColumn
                {
                    ColumnName = "UserName",
                }
            ],
            From = new SqlTableSource()
            {
                TableName = "customer",
            },
            Where = new SqlConditionExpression
            {
                Left = new SqlFieldExpression
                {
                    FieldName = "UserName",
                },
                ComparisonOperator = ComparisonOperator.Equal,
                Right = new SqlFieldExpression
                {
                    FieldName = "@username",
                }
            },
            OrderBy = new SqlOrderByClause
            {
                Columns = [
                    new SqlOrderByColumn
                    {
                        ColumnName = "LastDate",
                        Order = OrderType.Desc,
                    }
                ]
            }
        });
    }
    
    [Test]
    public void Many_SearchCondition()
    {
        var sql = $"""
                   select top (@batchsize) id from customer with(nolock) where matchid = @matchid and matchType < 39 and createDate <> @eventDate
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SelectStatement
        {
            Top = new SqlTopClause
            {
                Expression = new SqlGroup
                {
                    Inner = new SqlFieldExpression
                    {
                        FieldName = "@batchsize",
                    }
                },
            },
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
            Where = new SqlSearchCondition
            {
                Left = new SqlConditionExpression
                {
                    Left = new SqlFieldExpression
                    {
                        FieldName = "matchid"
                    },
                    ComparisonOperator = ComparisonOperator.Equal,
                    Right = new SqlFieldExpression()
                    {
                        FieldName = "@matchid"
                    }
                },
                LogicalOperator = LogicalOperator.And,
                Right = new SqlSearchCondition
                {
                    Left = new SqlConditionExpression
                    {
                        Left = new SqlFieldExpression
                        {
                            FieldName = "matchType"
                        },
                        ComparisonOperator = ComparisonOperator.LessThan,
                        Right = new SqlValue
                        {
                            SqlType = SqlType.IntValue,
                            Value = "39"
                        }
                    },
                    LogicalOperator = LogicalOperator.And,
                    Right = new SqlConditionExpression
                    {
                        Left = new SqlFieldExpression
                        {
                            FieldName = "createDate"
                        },
                        ComparisonOperator = ComparisonOperator.NotEqual,
                        Right = new SqlFieldExpression
                        {
                            FieldName = "@eventDate"
                        }
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
    public void Where_is_not_null()
    {
        var sql = $"""
                   select 1 
                   from customer
                   where createDate is not null
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SelectStatement
        {
            Columns = [
                new SelectColumn
                {
                    ColumnName = "1",
                }
            ],
            From = new SqlTableSource
            {
                TableName = "customer",
            },
            Where = new SqlConditionExpression
            {
                Left = new SqlFieldExpression
                {
                    FieldName = "createDate",
                },
                ComparisonOperator = ComparisonOperator.IsNot,
                Right = new SqlNullValue(),
            }
        });
    }

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

    private static ParseResult<ISqlExpression> ParseSql(string sql)
    {
        var p = new SqlParser(sql);
        return p.Parse();
    }
}