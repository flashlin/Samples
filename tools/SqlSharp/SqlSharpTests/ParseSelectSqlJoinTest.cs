using T1.SqlSharp.Expressions;

namespace SqlSharpTests;

[TestFixture]
public class ParseSelectSqlJoinTest
{
    [Test]
    public void Union_select_join_table_on()
    {
        var sql = $"""
                   select id from customer1 c
                   join emp e on c.date = e.date
                   union
                   select id from customer2
                   join emp2 on emp2.date = c.date
                   """;
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlFieldExpr { FieldName = "id" }
                }
            ],
            FromSources =
            [
                new SqlTableSource
                {
                    TableName = "customer1",
                    Alias = "c"
                },
                new SqlJoinTableCondition
                {
                    JoinType = JoinType.Inner,
                    JoinedTable = new SqlTableSource
                    {
                        TableName = "emp",
                        Alias = "e"
                    },
                    OnCondition = new SqlConditionExpression
                    {
                        Left = new SqlFieldExpr { FieldName = "c.date" },
                        ComparisonOperator = ComparisonOperator.Equal,
                        Right = new SqlFieldExpr { FieldName = "e.date" }
                    }
                }
            ],
            Unions =
            [
                new SqlUnionSelect
                {
                    SelectStatement = new SelectStatement
                    {
                        Columns =
                        [
                            new SelectColumn
                            {
                                Field = new SqlFieldExpr { FieldName = "id" }
                            }
                        ],
                        FromSources =
                        [
                            new SqlTableSource
                            {
                                TableName = "customer2"
                            },
                            new SqlJoinTableCondition
                            {
                                JoinType = JoinType.Inner,
                                JoinedTable = new SqlTableSource
                                {
                                    TableName = "emp2"
                                },
                                OnCondition = new SqlConditionExpression
                                {
                                    Left = new SqlFieldExpr { FieldName = "emp2.date" },
                                    ComparisonOperator = ComparisonOperator.Equal,
                                    Right = new SqlFieldExpr { FieldName = "c.date" }
                                }
                            }
                        ]
                    }
                }
            ]
        });
    }

    [Test]
    public void Select_Union_Select()
    {
        var sql = $"""
                   select id from customer
                   union
                   select id from emp
                   """;
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlFieldExpr { FieldName = "id" }
                }
            ],
            FromSources =
            [
                new SqlTableSource { TableName = "customer" }
            ],
            Unions =
            [
                new SqlUnionSelect
                {
                    SelectStatement = new SelectStatement
                    {
                        Columns =
                        [
                            new SelectColumn
                            {
                                Field = new SqlFieldExpr { FieldName = "id" }
                            }
                        ],
                        FromSources =
                        [
                            new SqlTableSource { TableName = "emp" }
                        ]
                    }
                }
            ]
        });
    }
    
    [Test]
    public void Join_group_select_from_inner_join_FROM()
    {
        var sql = $"""
                   select id
                   from @tempTable t 
                   join        
                   (        
                   	select id
                   	from emp b 
                   	inner join home h on b.field1 = e.field2, 
                   	@tb2 ts
                   	where b.id = 1
                   ) b on t.id = b.id
                   """;
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlFieldExpr { FieldName = "id" }
                }
            ],
            FromSources =
            [
                new SqlTableSource
                {
                    TableName = "@tempTable",
                    Alias = "t"
                },
                new SqlJoinTableCondition
                {
                    JoinType = JoinType.Inner,
                    JoinedTable = new SqlInnerTableSource
                    {
                        Inner = new SelectStatement
                        {
                            Columns =
                            [
                                new SelectColumn
                                {
                                    Field = new SqlFieldExpr { FieldName = "id" }
                                }
                            ],
                            FromSources =
                            [
                                new SqlTableSource { TableName = "emp", Alias = "b" },
                                new SqlJoinTableCondition
                                {
                                    JoinType = JoinType.Inner,
                                    JoinedTable = new SqlTableSource { TableName = "home", Alias = "h" },
                                    OnCondition = new SqlConditionExpression
                                    {
                                        Left = new SqlFieldExpr { FieldName = "b.field1" },
                                        ComparisonOperator = ComparisonOperator.Equal,
                                        Right = new SqlFieldExpr { FieldName = "e.field2" }
                                    }
                                },
                                new SqlTableSource { TableName = "@tb2", Alias = "ts" }
                            ],
                            Where = new SqlConditionExpression
                            {
                                Left = new SqlFieldExpr { FieldName = "b.id" },
                                ComparisonOperator = ComparisonOperator.Equal,
                                Right = new SqlValue
                                {
                                    SqlType = SqlType.IntValue, Value = "1"
                                }
                            }
                        },
                        Alias = "b"
                    },
                    OnCondition = new SqlConditionExpression
                    {
                        Left = new SqlFieldExpr { FieldName = "t.id" },
                        ComparisonOperator = ComparisonOperator.Equal,
                        Right = new SqlFieldExpr { FieldName = "b.id" }
                    }
                }
            ]
        });
    }
    
    [Test]
    public void From_group_select_inner_join()
    {
        var sql = $"""
                   SELECT id1
                   FROM (  
                    Select id2  
                    FROM (
                      SELECT id3 FROM customer
                    ) AS T  
                    inner join emp e on e.id = T.id3
                   ) as T2 
                   """;
        var rc = sql.ParseSql();
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlFieldExpr { FieldName = "id1" }
                }
            ],
            FromSources =
            [
                new SqlInnerTableSource
                {
                    Inner = new SqlParenthesizedExpression()
                    {
                        Inner = new SelectStatement
                        {
                            Columns =
                            [
                                new SelectColumn
                                {
                                    Field = new SqlFieldExpr { FieldName = "id2" }
                                }
                            ],
                            FromSources =
                            [
                                new SqlInnerTableSource
                                {
                                    Inner = new SqlParenthesizedExpression()
                                    {
                                        Inner = new SelectStatement
                                        {
                                            Columns =
                                            [
                                                new SelectColumn
                                                {
                                                    Field = new SqlFieldExpr { FieldName = "id3" }
                                                }
                                            ],
                                            FromSources =
                                            [
                                                new SqlTableSource { TableName = "customer" }
                                            ]
                                        }
                                    },
                                    Alias = "T"
                                },
                                new SqlJoinTableCondition
                                {
                                    JoinType = JoinType.Inner,
                                    JoinedTable = new SqlTableSource { TableName = "emp", Alias = "e" },
                                    OnCondition = new SqlConditionExpression
                                    {
                                        Left = new SqlFieldExpr { FieldName = "e.id" },
                                        ComparisonOperator = ComparisonOperator.Equal,
                                        Right = new SqlFieldExpr { FieldName = "T.id3" }
                                    }
                                }
                            ]
                        }
                    },
                    Alias = "T2"
                }
            ]
        });
    }
}