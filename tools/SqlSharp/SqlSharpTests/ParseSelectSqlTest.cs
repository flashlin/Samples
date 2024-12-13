using FluentAssertions;
using SqlSharpLit.Common.ParserLit;
using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;
using T1.Standard.DesignPatterns;

namespace SqlSharpTests;

[TestFixture]
public class ParseSelectSqlTest
{
    [Test]
    public void Select_union_select_join_table_on()
    {
        var sql = $"""
                   select id from customer1 c
                   join emp e on c.date = e.date
                   union
                   select id from customer2
                   join emp2 on emp2.date = c.date
                   """; 
        var rc = ParseSql(sql);
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
        var rc = ParseSql(sql);
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
    public void From_select_convert_datatype_field()
    {
        var sql = $"""
                   select *
                   from (
                   SELECT
                   	id,
                   	convert(nvarchar, customer.refno) refno
                     FROM customer
                   )
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlValue()
                    {
                        Value = "*"
                    }
                }
            ],
            FromSources =
            [
                new SqlInnerTableSource
                {
                    Inner = new SelectStatement
                    {
                        Columns =
                        [
                            new SelectColumn
                            {
                                Field = new SqlFieldExpr { FieldName = "id" }
                            },
                            new SelectColumn
                            {
                                Field = new SqlFunctionExpression
                                {
                                    FunctionName = "convert",
                                    Parameters =
                                    [
                                        new SqlDataType
                                        {
                                            DataTypeName = "nvarchar"
                                        },
                                        new SqlFieldExpr { FieldName = "customer.refno" }
                                    ]
                                },
                                Alias = "refno"
                            }
                        ],
                        FromSources = [new SqlTableSource { TableName = "customer" }]
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
        var rc = ParseSql(sql);
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
        var rc = ParseSql(sql);
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
    public void From_group_select()
    {
        var sql = $"""
                   select id from (select id from emp)
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SelectStatement()
        {
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlFieldExpr
                    {
                        FieldName = "id"
                    }
                }
            ],
            FromSources =
            [
                new SqlInnerTableSource()
                {
                    Inner = new SelectStatement()
                    {
                        Columns =
                        [
                            new SelectColumn
                            {
                                Field = new SqlFieldExpr
                                {
                                    FieldName = "id"
                                }
                            }
                        ],
                        FromSources =
                        [
                            new SqlTableSource
                            {
                                TableName = "emp"
                            }
                        ]
                    }
                }
            ]
        });
    }

    [Test]
    public void Where_id_in_select_from_custom_function()
    {
        var sql = $"""
                   SELECT id FROM customer
                   where id in (select val from [dbo].[strSplit](@text, ','))
                   """;
        var rc = ParseSql(sql);
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
                    TableName = "customer"
                }
            ],
            Where = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "id" },
                ComparisonOperator = ComparisonOperator.In,
                Right = new SqlGroup
                {
                    Inner = new SelectStatement
                    {
                        Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "val" } }],
                        FromSources =
                        [
                            new SqlFuncTableSource
                            {
                                Function = new SqlFunctionExpression
                                {
                                    FunctionName = "[dbo].[strSplit]",
                                    Parameters =
                                    [
                                        new SqlFieldExpr { FieldName = "@text" },
                                        new SqlValue { Value = "','" },
                                    ]
                                }
                            }
                        ]
                    }
                }
            }
        });
    }

    [Test]
    public void With_index()
    {
        var sql = $"""
                   select id
                   from customer with(nolock, INDEX (PK_customer))
                   """;
        var rc = ParseSql(sql);
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
                    Withs =
                    [
                        new SqlHint { Name = "nolock" },
                        new SqlTableHintIndex()
                        {
                            IndexValues = ["PK_customer"]
                        }
                    ]
                }
            ]
        });
    }

    [Test]
    public void Select_hex_value()
    {
        var sql = $"""
                   select 
                   ( CASE 
                   WHEN r = 0 THEN ( id & 0xFF00 )
                   END ) AS id
                   from customer
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlGroup
                    {
                        Inner = new SqlCaseExpr
                        {
                            WhenThens =
                            [
                                new SqlWhenThenClause
                                {
                                    When = new SqlConditionExpression
                                    {
                                        Left = new SqlFieldExpr { FieldName = "r" },
                                        ComparisonOperator = ComparisonOperator.Equal,
                                        Right = new SqlValue { SqlType = SqlType.IntValue, Value = "0" }
                                    },
                                    Then = new SqlGroup()
                                    {
                                        Inner = new SqlArithmeticBinaryExpr
                                        {
                                            Left = new SqlFieldExpr { FieldName = "id" },
                                            Operator = ArithmeticOperator.BitwiseAnd,
                                            Right = new SqlValue { SqlType = SqlType.HexValue, Value = "0xFF00" }
                                        },
                                    }
                                }
                            ]
                        }
                    },
                    Alias = "id"
                }
            ],
            FromSources =
            [
                new SqlTableSource { TableName = "customer" }
            ]
        });
    }

    [Test]
    public void Where_expr_and_group()
    {
        var sql = $"""
                   SELECT id
                   FROM customer
                   WHERE id = 1 
                   AND (@code is null or [code] = @Code)
                   """;
        var rc = ParseSql(sql);
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
            Where = new SqlSearchCondition
            {
                Left = new SqlConditionExpression
                {
                    Left = new SqlFieldExpr { FieldName = "id" },
                    ComparisonOperator = ComparisonOperator.Equal,
                    Right = new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
                },
                LogicalOperator = LogicalOperator.And,
                Right = new SqlGroup()
                {
                    Inner = new SqlSearchCondition()
                    {
                        Left = new SqlConditionExpression
                        {
                            Left = new SqlFieldExpr { FieldName = "@code" },
                            ComparisonOperator = ComparisonOperator.Is,
                            Right = new SqlNullValue()
                        },
                        LogicalOperator = LogicalOperator.Or,
                        Right = new SqlConditionExpression
                        {
                            Left = new SqlFieldExpr { FieldName = "[code]" },
                            ComparisonOperator = ComparisonOperator.Equal,
                            Right = new SqlFieldExpr { FieldName = "@Code" }
                        }
                    }
                }
            }
        });
    }

    [Test]
    public void Select_count_distinct()
    {
        var sql = $"""
                   SELECT
                   	 COUNT(DISTINCT id) AS UserCount
                   FROM customer
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlFunctionExpression
                    {
                        FunctionName = "COUNT",
                        Parameters =
                        [
                            new SqlDistinct()
                            {
                                Value = new SqlFieldExpr
                                {
                                    FieldName = "id"
                                },
                            }
                        ]
                    },
                    Alias = "UserCount"
                }
            ],
            FromSources =
            [
                new SqlTableSource { TableName = "customer" }
            ]
        });
    }


    [Test]
    public void Where_grant_equal()
    {
        var sql = $"""
                   select id from customer
                   where id > = 1
                   """;
        var rc = ParseSql(sql);
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
            Where = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "id" },
                ComparisonOperator = ComparisonOperator.GreaterThanOrEqual,
                Right = new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
            }
        });
    }

    [Test]
    public void From_table_aliasTableName()
    {
        var sql = $"""
                   select c.id from customer c
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlFieldExpr { FieldName = "c.id" }
                }
            ],
            FromSources =
            [
                new SqlTableSource { TableName = "customer", Alias = "c" }
            ]
        });
    }

    [Test]
    public void Select_select_table_tableAliasName_with()
    {
        var sql = $"""
                   select 
                   (select id from customer e with(nolock)) as id2
                   from customer c 
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlGroup
                    {
                        Inner = new SelectStatement
                        {
                            Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "id" } }],
                            FromSources =
                            [
                                new SqlTableSource
                                {
                                    TableName = "customer",
                                    Alias = "e",
                                    Withs = [new SqlHint { Name = "nolock" }]
                                }
                            ]
                        }
                    },
                    Alias = "id2"
                }
            ],
            FromSources =
            [
                new SqlTableSource { TableName = "customer", Alias = "c" }
            ]
        });
    }

    [Test]
    public void Select_other_select_from_table_aliasTableName_aliasFieldName()
    {
        var sql = $"""
                   select 
                   (select e.id from emp e) as id1 
                   from customer
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlGroup
                    {
                        Inner = new SelectStatement
                        {
                            Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "e.id" } }],
                            FromSources =
                            [
                                new SqlTableSource { TableName = "emp", Alias = "e" }
                            ]
                        }
                    },
                    Alias = "id1"
                }
            ],
            FromSources =
            [
                new SqlTableSource { TableName = "customer" }
            ]
        });
    }


    [Test]
    public void Where_between()
    {
        var sql = $"""
                   select id from customer
                   where id between 1 and 10
                   """;
        var rc = ParseSql(sql);
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
            Where = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "id" },
                ComparisonOperator = ComparisonOperator.Between,
                Right = new SqlBetweenValue
                {
                    Start = new SqlValue { SqlType = SqlType.IntValue, Value = "1" },
                    End = new SqlValue { SqlType = SqlType.IntValue, Value = "10" }
                }
            }
        });
    }

    [Test]
    public void Case_when_then_negative_number()
    {
        var sql = $"""
                   select (case when n=0 and id > 0 then -id * 100 else 1 end)
                   """;
        var rc = ParseSql(sql);
        var selectStatement = (SelectStatement)rc.ResultValue;
        var groupInner = ((SqlGroup)selectStatement.Columns[0].Field).Inner;
        var field = (SqlCaseExpr)groupInner;
        var whenThen0 = field.WhenThens[0];
        whenThen0.When.ShouldBe(new SqlSearchCondition()
            {
                Left = new SqlConditionExpression
                {
                    Left = new SqlFieldExpr { FieldName = "n" },
                    ComparisonOperator = ComparisonOperator.Equal,
                    Right = new SqlValue { SqlType = SqlType.IntValue, Value = "0" }
                },
                LogicalOperator = LogicalOperator.And,
                Right = new SqlConditionExpression
                {
                    Left = new SqlFieldExpr { FieldName = "id" },
                    ComparisonOperator = ComparisonOperator.GreaterThan,
                    Right = new SqlValue { SqlType = SqlType.IntValue, Value = "0" }
                }
            }
        );
        whenThen0.Then.ShouldBe(
            new SqlArithmeticBinaryExpr
            {
                Left = new SqlNegativeValue()
                {
                    Value = new SqlFieldExpr { FieldName = "id" },
                },
                Operator = ArithmeticOperator.Multiply,
                Right = new SqlValue { SqlType = SqlType.IntValue, Value = "100" }
            }
        );
    }

    [Test]
    public void select_other_select()
    {
        var sql = $"""
                   SELECT id, 
                    (SELECT @name) AS name
                   from customer 
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlFieldExpr { FieldName = "id" }
                },
                new SelectColumn
                {
                    Field = new SqlGroup
                    {
                        Inner = new SelectStatement
                        {
                            Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "@name" } }]
                        }
                    },
                    Alias = "name"
                }
            ],
            FromSources =
            [
                new SqlTableSource { TableName = "customer" }
            ]
        });
    }

    [Test]
    public void where_field_in_other_select()
    {
        var sql = $"""
                   select id
                   from customer
                   where gid in (select gid from temp)
                   """;
        var rc = ParseSql(sql);
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
                    TableName = "customer"
                }
            ],
            Where = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "gid" },
                ComparisonOperator = ComparisonOperator.In,
                Right = new SqlGroup
                {
                    Inner = new SelectStatement
                    {
                        Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "gid" } }],
                        FromSources =
                        [
                            new SqlTableSource { TableName = "temp" }
                        ]
                    }
                }
            }
        });
    }


    [Test]
    public void case_has_when()
    {
        var sql = $"""
                   select @id = case id&2 when 2 then 1 else 0 end
                   from customer
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlAssignExpr
                    {
                        Left = new SqlFieldExpr { FieldName = "@id" },
                        Right = new SqlCaseExpr
                        {
                            When = new SqlArithmeticBinaryExpr
                            {
                                Left = new SqlFieldExpr { FieldName = "id" },
                                Operator = ArithmeticOperator.BitwiseAnd,
                                Right = new SqlValue { SqlType = SqlType.IntValue, Value = "2" }
                            },
                            WhenThens =
                            [
                                new SqlWhenThenClause
                                {
                                    When = new SqlValue { SqlType = SqlType.IntValue, Value = "2" },
                                    Then = new SqlValue { SqlType = SqlType.IntValue, Value = "1" }
                                }
                            ],
                            Else = new SqlValue { SqlType = SqlType.IntValue, Value = "0" }
                        }
                    }
                }
            ],
            FromSources =
            [
                new SqlTableSource { TableName = "customer" }
            ]
        });
    }

    [Test]
    public void select_case()
    {
        var sql = $"""
                   SELECT id,
                          ( CASE 
                              WHEN id IN (1, 2) THEN 100
                              ELSE 200
                            END ) AS balance
                   from customer
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlFieldExpr()
                    {
                        FieldName = "id"
                    },
                },
                new SelectColumn
                {
                    Field = new SqlGroup
                    {
                        Inner = new SqlCaseExpr
                        {
                            WhenThens =
                            [
                                new SqlWhenThenClause
                                {
                                    When = new SqlConditionExpression
                                    {
                                        Left = new SqlFieldExpr
                                        {
                                            FieldName = "id"
                                        },
                                        ComparisonOperator = ComparisonOperator.In,
                                        Right = new SqlValues
                                        {
                                            Items =
                                            [
                                                new SqlValue
                                                {
                                                    SqlType = SqlType.IntValue,
                                                    Value = "1"
                                                },
                                                new SqlValue
                                                {
                                                    SqlType = SqlType.IntValue,
                                                    Value = "2"
                                                }
                                            ]
                                        }
                                    },
                                    Then = new SqlValue
                                    {
                                        SqlType = SqlType.IntValue,
                                        Value = "100"
                                    }
                                }
                            ],
                            Else = new SqlValue
                            {
                                SqlType = SqlType.IntValue,
                                Value = "200"
                            }
                        }
                    },
                    Alias = "balance"
                }
            ],
            FromSources =
            [
                new SqlTableSource
                {
                    TableName = "customer"
                }
            ]
        });
    }

    [Test]
    public void where_group_and_group()
    {
        var sql = $"""
                   select  id
                   from customer 
                   where 
                   		(IsNull(name, 0) <15)and 
                   		(id1!=0 or id2!=0)
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SelectStatement()
        {
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlFieldExpr
                    {
                        FieldName = "id"
                    },
                }
            ],
            FromSources =
            [
                new SqlTableSource
                {
                    TableName = "customer",
                }
            ],
            Where = new SqlSearchCondition
            {
                Left = new SqlGroup
                {
                    Inner = new SqlConditionExpression
                    {
                        Left = new SqlFunctionExpression
                        {
                            FunctionName = "IsNull",
                            Parameters =
                            [
                                new SqlFieldExpr
                                {
                                    FieldName = "name",
                                },
                                new SqlValue
                                {
                                    SqlType = SqlType.IntValue,
                                    Value = "0"
                                }
                            ]
                        },
                        ComparisonOperator = ComparisonOperator.LessThan,
                        Right = new SqlValue
                        {
                            SqlType = SqlType.IntValue,
                            Value = "15"
                        }
                    }
                },
                LogicalOperator = LogicalOperator.And,
                Right = new SqlGroup
                {
                    Inner = new SqlSearchCondition
                    {
                        Left = new SqlConditionExpression
                        {
                            Left = new SqlFieldExpr
                            {
                                FieldName = "id1"
                            },
                            ComparisonOperator = ComparisonOperator.NotEqual,
                            Right = new SqlValue
                            {
                                SqlType = SqlType.IntValue,
                                Value = "0"
                            }
                        },
                        LogicalOperator = LogicalOperator.Or,
                        Right = new SqlConditionExpression
                        {
                            Left = new SqlFieldExpr
                            {
                                FieldName = "id2"
                            },
                            ComparisonOperator = ComparisonOperator.NotEqual,
                            Right = new SqlValue
                            {
                                SqlType = SqlType.IntValue,
                                Value = "0"
                            }
                        }
                    }
                }
            }
        });
    }

    [Test]
    public void where_group_func_lessThan_int()
    {
        var sql = $"""
                   select id from customer
                   where (IsNull(Status, 0) <15)
                   """;
        var rc = ParseSql(sql);
        var json = rc.ResultValue.ToSqlJsonString();
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlFieldExpr()
                    {
                        FieldName = "id"
                    },
                }
            ],
            FromSources =
            [
                new SqlTableSource
                {
                    TableName = "customer",
                }
            ],
            Where = new SqlGroup()
            {
                Inner = new SqlConditionExpression
                {
                    Left = new SqlFunctionExpression
                    {
                        FunctionName = "IsNull",
                        Parameters =
                        [
                            new SqlFieldExpr
                            {
                                FieldName = "Status",
                            },
                            new SqlValue
                            {
                                SqlType = SqlType.IntValue,
                                Value = "0"
                            }
                        ]
                    },
                    ComparisonOperator = ComparisonOperator.LessThan,
                    Right = new SqlValue
                    {
                        SqlType = SqlType.IntValue,
                        Value = "15"
                    }
                }
            }
        });
    }

    [Test]
    public void select_variable_as_field()
    {
        var sql = $"""
                   select
                   @a as 'field1',
                   @b as 'field2'
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlFieldExpr
                    {
                        FieldName = "@a"
                    },
                    Alias = "'field1'",
                },
                new SelectColumn
                {
                    Field = new SqlFieldExpr
                    {
                        FieldName = "@b"
                    },
                    Alias = "'field2'",
                },
            ]
        });
    }

    [Test]
    public void Select_with_expr1_and_expr2_and_expr3()
    {
        var sql = $"""
                   SELECT TOP (@batchsize) Id
                   FROM customer
                   WHERE [status] & (123 | 456) = 0
                       AND ISNULL(pp,0) = 0
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SelectStatement
        {
            Top = new SqlTopClause
            {
                Expression = new SqlGroup
                {
                    Inner = new SqlFieldExpr
                    {
                        FieldName = "@batchsize",
                    }
                },
            },
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlFieldExpr
                    {
                        FieldName = "Id"
                    },
                }
            ],
            FromSources =
            [
                new SqlTableSource
                {
                    TableName = "customer",
                }
            ],
            Where = new SqlSearchCondition
            {
                Left = new SqlConditionExpression
                {
                    Left = new SqlArithmeticBinaryExpr
                    {
                        Left = new SqlFieldExpr
                        {
                            FieldName = "[status]",
                        },
                        Operator = ArithmeticOperator.BitwiseAnd,
                        Right = new SqlGroup()
                        {
                            Inner = new SqlArithmeticBinaryExpr
                            {
                                Left = new SqlValue
                                {
                                    SqlType = SqlType.IntValue,
                                    Value = "123",
                                },
                                Operator = ArithmeticOperator.BitwiseOr,
                                Right = new SqlValue
                                {
                                    SqlType = SqlType.IntValue,
                                    Value = "456",
                                }
                            }
                        }
                    },
                    ComparisonOperator = ComparisonOperator.Equal,
                    Right = new SqlValue
                    {
                        SqlType = SqlType.IntValue,
                        Value = "0",
                    }
                },
                LogicalOperator = LogicalOperator.And,
                Right = new SqlConditionExpression
                {
                    Left = new SqlFunctionExpression
                    {
                        FunctionName = "ISNULL",
                        Parameters =
                        [
                            new SqlFieldExpr
                            {
                                FieldName = "pp",
                            },
                            new SqlValue
                            {
                                SqlType = SqlType.IntValue,
                                Value = "0",
                            }
                        ]
                    },
                    ComparisonOperator = ComparisonOperator.Equal,
                    Right = new SqlValue
                    {
                        SqlType = SqlType.IntValue,
                        Value = "0",
                    }
                }
            }
        });
    }

    [Test]
    public void Select_Field_Equal_Func_Arithmetic()
    {
        var sql = $"""
                   select @a = round(((@b - @b) * RandomNumber + @d), 0)
                   from customer
                   """;
        var rc = ParseSql(sql);
    }

    [Test]
    public void From_Select()
    {
        var sql = $"""
                   select @random = round(((@upper - @lower -1) * RandomNumber + @lower), 0)
                   from (select RandomNumber from vRandNumber) as d
                   """;
    }

    [Test]
    public void Select_Field_equal_count_star()
    {
        var sql = $"""
                   select @upper = count(*) 
                   from customer
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlAssignExpr
                    {
                        Left = new SqlFieldExpr
                        {
                            FieldName = "@upper"
                        },
                        Right = new SqlFunctionExpression
                        {
                            FunctionName = "count",
                            Parameters =
                            [
                                new SqlValue
                                {
                                    Value = "*",
                                }
                            ]
                        }
                    },
                }
            ],
            FromSources =
            [
                new SqlTableSource
                {
                    TableName = "customer",
                }
            ]
        });
    }

    [Test]
    public void Select_field_equal_binaryExpr()
    {
        var sql = $"""
                   select @a = @a & b
                   from customer 
                   """;
        var rc = ParseSql(sql);
        var json = rc.ResultValue.ToSqlJsonString();
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn()
                {
                    Field = new SqlAssignExpr
                    {
                        Left = new SqlFieldExpr()
                        {
                            FieldName = "@a",
                        },
                        Right = new SqlArithmeticBinaryExpr
                        {
                            Left = new SqlFieldExpr
                            {
                                FieldName = "@a",
                            },
                            Operator = ArithmeticOperator.BitwiseAnd,
                            Right = new SqlFieldExpr()
                            {
                                FieldName = "b",
                            }
                        }
                    }
                },
            ],
            FromSources =
            [
                new SqlTableSource
                {
                    TableName = "customer",
                }
            ]
        });
    }

    [Test]
    public void Where_field_in()
    {
        var sql = $"""
                   select @a
                   from customer 
                   where id in (@b, @c)
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SelectStatement()
        {
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlFieldExpr()
                    {
                        FieldName = "@a"
                    },
                }
            ],
            FromSources =
            [
                new SqlTableSource
                {
                    TableName = "customer",
                }
            ],
            Where = new SqlConditionExpression
            {
                Left = new SqlFieldExpr
                {
                    FieldName = "id",
                },
                ComparisonOperator = ComparisonOperator.In,
                Right = new SqlValues
                {
                    Items =
                    [
                        new SqlFieldExpr
                        {
                            FieldName = "@b",
                        },
                        new SqlFieldExpr
                        {
                            FieldName = "@c",
                        },
                    ]
                }
            }
        });
    }

    [Test]
    public void Select_field_as()
    {
        var sql = $"""
                   select -1 as [result]
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlValue
                    {
                        SqlType = SqlType.IntValue,
                        Value = "-1",
                    },
                    Alias = "[result]",
                }
            ]
        });
    }

    [Test]
    public void Where_cast_plus_cast()
    {
        var sql = $"""
                   select id from customer 
                   where BetOption = 
                   cast(@a as nvarchar(2)) + ':' + cast(@b as nvarchar(3))
                   """;
        var rc = ParseSql(sql);
        var json = rc.ResultValue.ToSqlJsonString();
        rc.ShouldBe(new SelectStatement()
        {
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlFieldExpr()
                    {
                        FieldName = "id"
                    },
                }
            ],
            FromSources =
            [
                new SqlTableSource
                {
                    TableName = "customer",
                }
            ],
            Where = new SqlConditionExpression
            {
                Left = new SqlFieldExpr
                {
                    FieldName = "BetOption",
                },
                ComparisonOperator = ComparisonOperator.Equal,
                Right = new SqlArithmeticBinaryExpr
                {
                    Left = new SqlArithmeticBinaryExpr()
                    {
                        Left = new SqlFunctionExpression()
                        {
                            FunctionName = "cast",
                            Parameters =
                            [
                                new SqlAsExpr
                                {
                                    Instance = new SqlFieldExpr
                                    {
                                        FieldName = "@a",
                                    },
                                    As = new SqlDataType
                                    {
                                        DataTypeName = "nvarchar",
                                        Size = new SqlDataSize
                                        {
                                            Size = "2",
                                        }
                                    }
                                }
                            ]
                        },
                        Operator = ArithmeticOperator.Add,
                        Right = new SqlValue
                        {
                            Value = "':'",
                        }
                    },
                    Operator = ArithmeticOperator.Add,
                    Right = new SqlFunctionExpression
                    {
                        FunctionName = "cast",
                        Parameters =
                        [
                            new SqlAsExpr
                            {
                                Instance = new SqlFieldExpr
                                {
                                    FieldName = "@b",
                                },
                                As = new SqlDataType
                                {
                                    DataTypeName = "nvarchar",
                                    Size = new SqlDataSize
                                    {
                                        Size = "3",
                                    }
                                }
                            }
                        ]
                    }
                },
            }
        });
    }

    [Test]
    public void Select_Star()
    {
        var sql = $"""
                   SELECT * FROM [$(TableName)]
                   """;
        var rc = ParseSql(sql);
        rc.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlValue() { Value = "*" },
                }
            ],
            FromSources =
            [
                new SqlTableSource
                {
                    TableName = "[$(TableName)]",
                }
            ]
        });
    }

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
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlFieldExpr()
                    {
                        FieldName = "UserName"
                    },
                }
            ],
            FromSources =
            [
                new SqlTableSource()
                {
                    TableName = "customer",
                }
            ],
            Where = new SqlConditionExpression
            {
                Left = new SqlFieldExpr
                {
                    FieldName = "UserName",
                },
                ComparisonOperator = ComparisonOperator.Equal,
                Right = new SqlFieldExpr
                {
                    FieldName = "@username",
                }
            },
            OrderBy = new SqlOrderByClause
            {
                Columns =
                [
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
        var sqlJson = rc.ResultValue.ToSqlJsonString();
        rc.ShouldBe(new SelectStatement
        {
            Top = new SqlTopClause
            {
                Expression = new SqlGroup
                {
                    Inner = new SqlFieldExpr
                    {
                        FieldName = "@batchsize",
                    }
                },
            },
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlFieldExpr()
                    {
                        FieldName = "id"
                    },
                }
            ],
            FromSources =
            [
                new SqlTableSource
                {
                    TableName = "customer",
                    Withs =
                    [
                        new SqlHint()
                        {
                            Name = "nolock",
                        }
                    ],
                }
            ],
            Where = new SqlSearchCondition
            {
                Left = new SqlSearchCondition
                {
                    Left = new SqlConditionExpression
                    {
                        Left = new SqlFieldExpr
                        {
                            FieldName = "matchid"
                        },
                        ComparisonOperator = ComparisonOperator.Equal,
                        Right = new SqlFieldExpr()
                        {
                            FieldName = "@matchid"
                        }
                    },
                    LogicalOperator = LogicalOperator.And,
                    Right = new SqlConditionExpression
                    {
                        Left = new SqlFieldExpr
                        {
                            FieldName = "matchType"
                        },
                        ComparisonOperator = ComparisonOperator.LessThan,
                        Right = new SqlValue
                        {
                            SqlType = SqlType.IntValue,
                            Value = "39"
                        }
                    }
                },
                LogicalOperator = LogicalOperator.And,
                Right = new SqlConditionExpression
                {
                    Left = new SqlFieldExpr
                    {
                        FieldName = "createDate"
                    },
                    ComparisonOperator = ComparisonOperator.NotEqual,
                    Right = new SqlFieldExpr
                    {
                        FieldName = "@eventDate"
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
                new SelectColumn
                {
                    Field = new SqlFieldExpr
                    {
                        FieldName = "Id"
                    }
                },
                new SelectColumn
                {
                    Field = new SqlFieldExpr()
                    {
                        FieldName = "Name"
                    }
                },
            ],
            FromSources =
            [
                new SqlTableSource
                {
                    TableName = "Persons"
                }
            ],
            Where = new SqlConditionExpression
            {
                Left = new SqlFieldExpr
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
                new SelectColumn
                {
                    Field = new SqlValue
                    {
                        SqlType = SqlType.IntValue,
                        Value = "1"
                    },
                },
            ],
            FromSources =
            [
                new SqlTableSource
                {
                    TableName = "sys.databases",
                }
            ],
            Where = new SqlSearchCondition()
            {
                Left = new SqlConditionExpression
                {
                    Left = new SqlFieldExpr
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
                            new SqlFieldExpr
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
            Columns =
            [
                new SelectColumn()
                {
                    Field = new SqlValue
                    {
                        SqlType = SqlType.IntValue,
                        Value = "1",
                    }
                }
            ],
            FromSources =
            [
                new SqlTableSource
                {
                    TableName = "customer",
                }
            ],
            Where = new SqlConditionExpression
            {
                Left = new SqlFieldExpr
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
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlFieldExpr
                    {
                        FieldName = "id"
                    },
                }
            ],
            FromSources =
            [
                new SqlTableSource
                {
                    TableName = "customer",
                    Withs =
                    [
                        new SqlHint()
                        {
                            Name = "nolock",
                        }
                    ],
                }
            ],
        });
    }

    private static ParseResult<ISqlExpression> ParseSql(string sql)
    {
        var p = new SqlParser(sql);
        return p.Parse();
    }
}