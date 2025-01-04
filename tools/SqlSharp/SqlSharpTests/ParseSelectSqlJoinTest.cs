using T1.SqlSharp.Expressions;

namespace SqlSharpTests;

[TestFixture]
public class ParseSelectSqlJoinTest
{
    [Test]
    public void inner_join_func()
    {
        var sql = $"""
                   select id
                   from customer c
                   inner join StrSplitMax(@customerIDs, ',') t
                   on c.id = t.val
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
                new SqlTableSource { TableName = "customer", Alias = "c" },
                new SqlJoinTableCondition
                {
                    JoinType = JoinType.Inner,
                    JoinedTable = new SqlFuncTableSource
                    {
                        Function = new SqlFunctionExpression()
                        {
                            FunctionName = "StrSplitMax",
                            Parameters =
                            [
                                new SqlFieldExpr {  FieldName= "@customerIDs" },
                                new SqlValue { Value = "','" }
                            ]
                        },
                        Alias = "t"
                    },
                    OnCondition = new SqlConditionExpression
                    {
                        Left = new SqlFieldExpr { FieldName = "c.id" },
                        ComparisonOperator = ComparisonOperator.Equal,
                        Right = new SqlFieldExpr { FieldName = "t.val" }
                    },
                }
            ]
        });
    }
    
    [Test]
    public void Left_outer_join()
    {
        var sql = $"""
                   select id 
                   from customer c
                   left outer join emp e on e.id = c.id  
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
                new SqlTableSource { TableName = "customer", Alias = "c" },
                new SqlJoinTableCondition
                {
                    JoinType = JoinType.Left,
                    JoinedTable = new SqlTableSource { TableName = "emp", Alias = "e" },
                    OnCondition = new SqlConditionExpression
                    {
                        Left = new SqlFieldExpr { FieldName = "e.id" },
                        ComparisonOperator = ComparisonOperator.Equal,
                        Right = new SqlFieldExpr { FieldName = "c.id" }
                    }
                }
            ]
        });
    }

    [Test]
    public void Right_outer_join()
    {
        var sql = $"""
                   select id
                   from customer c
                   right outer join tb on tb.id = c.id  
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
                new SqlTableSource { TableName = "customer", Alias = "c" },
                new SqlJoinTableCondition
                {
                    JoinType = JoinType.Right,
                    JoinedTable = new SqlTableSource { TableName = "tb" },
                    OnCondition = new SqlConditionExpression
                    {
                        Left = new SqlFieldExpr { FieldName = "tb.id" },
                        ComparisonOperator = ComparisonOperator.Equal,
                        Right = new SqlFieldExpr { FieldName = "c.id" }
                    }
                },
            ]
        });
    }
    
    [Test]
    public void From_group_table_join()
    {
        var sql = $"""
                   SELECT id
                   FROM (
                     customer c 
                     join emp e on c.id = e.id
                   )
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
                new SqlTableSource { TableName = "customer", Alias = "c" },
                new SqlJoinTableCondition
                {
                    JoinType = JoinType.Inner,
                    JoinedTable = new SqlTableSource { TableName = "emp", Alias = "e" },
                    OnCondition = new SqlConditionExpression
                    {
                        Left = new SqlFieldExpr { FieldName = "c.id" },
                        ComparisonOperator = ComparisonOperator.Equal,
                        Right = new SqlFieldExpr { FieldName = "e.id" }
                    }
                }
            ]
        });
    }
    
    [Test]
    public void LeftJoin_group()
    {
        var sql = $"""
                   select id
                   from	customer c
                   left join (select id from emp) e on e.id = c.id
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
                new SqlTableSource { TableName = "customer", Alias = "c" },
                new SqlJoinTableCondition
                {
                    JoinType = JoinType.Left,
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
                                new SqlTableSource { TableName = "emp" }
                            ],
                        },
                        Alias = "e"
                    },
                    OnCondition = new SqlConditionExpression
                    {
                        Left = new SqlFieldExpr { FieldName = "e.id" },
                        ComparisonOperator = ComparisonOperator.Equal,
                        Right = new SqlFieldExpr { FieldName = "c.id" }
                    }
                }
            ]
        });
    }
    
    [Test]
    public void InnerJoin()
    {
        var sql = $"""
                   SELECT id
                   FROM customer c
                   INNER JOIN emp e
                   	ON c.id = e.id
                   	AND c.name = '123'
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
                    TableName = "customer",
                    Alias = "c",
                },
                new SqlJoinTableCondition
                {
                    JoinType = JoinType.Inner,
                    JoinedTable = new SqlTableSource
                    {
                        TableName = "emp",
                        Alias = "e"
                    },
                    OnCondition = new SqlSearchCondition
                    {
                        Left = new SqlConditionExpression
                        {
                            Left = new SqlFieldExpr { FieldName = "c.id" },
                            ComparisonOperator = ComparisonOperator.Equal,
                            Right = new SqlFieldExpr { FieldName = "e.id" }
                        },
                        LogicalOperator = LogicalOperator.And,
                        Right = new SqlConditionExpression
                        {
                            Left = new SqlFieldExpr { FieldName = "c.name" },
                            ComparisonOperator = ComparisonOperator.Equal,
                            Right = new SqlValue { Value = "'123'" }
                        }
                    }
                }
            ]
        });
    }

    [Test]
    public void Join_table()
    {
        var sql = $"""
                   select id from customer c 
                   join [order] o on c.id = o.customerId
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
                    TableName = "customer",
                    Alias = "c",
                },
                new SqlJoinTableCondition
                {
                    JoinType = JoinType.Inner,
                    JoinedTable = new SqlTableSource
                    {
                        TableName = "[order]",
                        Alias = "o"
                    },
                    OnCondition = new SqlConditionExpression
                    {
                        Left = new SqlFieldExpr { FieldName = "c.id" },
                        ComparisonOperator = ComparisonOperator.Equal,
                        Right = new SqlFieldExpr { FieldName = "o.customerId" }
                    }
                }
            ],
        });
    }
    
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