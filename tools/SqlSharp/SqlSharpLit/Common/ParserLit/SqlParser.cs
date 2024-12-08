using System.Diagnostics;
using System.Runtime.InteropServices.JavaScript;
using System.Text.RegularExpressions;
using T1.SqlSharp.Expressions;

namespace SqlSharpLit.Common.ParserLit;

public class SqlParser
{
    private const string ConstraintKeyword = "CONSTRAINT";

    private readonly StringParser _text;

    public SqlParser(string text)
    {
        _text = new StringParser(text);
    }

    public IEnumerable<ISqlExpression> Extract()
    {
        while (!_text.IsEnd())
        {
            var rc = Parse();
            if (rc.HasValue)
            {
                yield return rc.ResultValue;
            }
            else
            {
                _text.ReadUntil(c => c == '\n');
            }
        }
    }

    public static ParseResult<ISqlExpression> Parse(string sql)
    {
        var p = new SqlParser(sql);
        return p.Parse();
    }

    public ParseResult<ISqlExpression> Parse()
    {
        if (Try(ParseCreateTableStatement, out var createTableResult))
        {
            return createTableResult.Result;
        }

        if (Try(ParseSelectStatement, out var selectResult))
        {
            return selectResult.Result;
        }

        if (Try(ParseExecSpAddExtendedProperty, out var execSpAddExtendedPropertyResult))
        {
            return execSpAddExtendedPropertyResult.Result;
        }

        return CreateParseError("Unknown statement");
    }


    public ParseResult<List<ISqlExpression>> ParseCreateTableColumns()
    {
        var columns = new List<ISqlExpression>();
        do
        {
            SkipWhiteSpace();

            // 一開始就定義這些 關鍵字表示不是 column  
            if (IsAny(PeekKeywords("CONSTRAINT"), PeekKeywords("PRIMARY", "KEY"), PeekKeywords("UNIQUE"),
                    PeekKeywords("FOREIGN", "KEY")))
            {
                break;
            }

            var columnDefinition = Or<ISqlExpression>(ParseComputedColumnDefinition, ParseColumnDefinition)();
            if (columnDefinition.HasError)
            {
                return columnDefinition.Error;
            }

            if (columnDefinition.Result != null)
            {
                columns.Add(columnDefinition.Result);
            }

            if (_text.PeekChar() != ',')
            {
                break;
            }

            _text.ReadChar();

            // 怪異的 SQL 語法: 允許逗號後面沒有東西 遇到 ) 直接結束 
            if (_text.PeekChar() == ')')
            {
                break;
            }
        } while (!_text.IsEnd());

        return CreateParseResult(columns);
    }

    public ParseResult<SqlCreateTableExpression> ParseCreateTableStatement()
    {
        if (!TryKeywords("CREATE", "TABLE"))
        {
            return NoneResult<SqlCreateTableExpression>();
        }

        var tableName = _text.ReadSqlIdentifier();
        if (!TryMatch("("))
        {
            return CreateParseError("Expected (");
        }

        var createTableStatement = new SqlCreateTableExpression()
        {
            TableName = tableName.Word,
        };

        while (!_text.IsEnd())
        {
            var tableColumnsResult = ParseCreateTableColumns();
            if (tableColumnsResult.HasError)
            {
                return tableColumnsResult.Error;
            }

            var tableColumns = tableColumnsResult.ResultValue;
            if (tableColumns.Count > 0)
            {
                createTableStatement.Columns.AddRange(tableColumns);
                continue;
            }

            var tableConstraintsResult = ParseWithComma(ParseTableConstraint);
            if (tableConstraintsResult.HasError)
            {
                return tableConstraintsResult.Error;
            }

            var tableConstraints = tableConstraintsResult.ResultValue;
            if (tableConstraints.Count > 0)
            {
                createTableStatement.Constraints.AddRange(tableConstraints);
                continue;
            }

            break;
        }

        if (!TryMatch(")"))
        {
            return CreateParseError("ParseCreateTableStatement Expected )");
        }

        SkipStatementEnd();

        return CreateParseResult(createTableStatement);
    }

    public ParseResult<SqlSpAddExtendedPropertyExpression> ParseExecSpAddExtendedProperty()
    {
        if (!TryKeywords("EXEC", "SP_AddExtendedProperty"))
        {
            return NoneResult<SqlSpAddExtendedPropertyExpression>();
        }

        var parameters = ParseWithComma(ParseParameterValueOrAssignValue);
        if (parameters.HasError)
        {
            return parameters.Error;
        }

        if (parameters.ResultValue.Count != 8)
        {
            return CreateParseError("Expected 8 parameters");
        }

        var p = parameters.ResultValue;

        var sqlSpAddExtendedProperty = new SqlSpAddExtendedPropertyExpression
        {
            Name = p[0].Value,
            Value = p[1].Value,
            Level0Type = p[2].Value,
            Level0Name = p[3].Value,
            Level1Type = p[4].Value,
            Level1Name = p[5].Value,
            Level2Type = p[6].Value,
            Level2Name = p[7].Value
        };
        return CreateParseResult(sqlSpAddExtendedProperty);
    }

    public ParseResult<SqlConstraintForeignKey> ParseForeignKeyExpression()
    {
        if (!TryKeywords("FOREIGN", "KEY"))
        {
            return NoneResult<SqlConstraintForeignKey>();
        }

        var columnsResult = ParseColumnsAscDesc();
        if (columnsResult.HasError)
        {
            return columnsResult.Error;
        }

        var columns = columnsResult.ResultValue;
        if (!TryKeyword("REFERENCES"))
        {
            return CreateParseError("Expected REFERENCES");
        }

        var tableName = _text.ReadSqlIdentifier();
        if (tableName.Length == 0)
        {
            return CreateParseError("Expected reference table name");
        }

        var refColumn = string.Empty;
        if (TryMatch("("))
        {
            refColumn = _text.ReadSqlIdentifier().Word;
            if (!TryMatch(")"))
            {
                return CreateParseError("Expected )");
            }
        }

        var onDelete = ReferentialAction.NoAction;
        if (TryKeywords("ON", "DELETE"))
        {
            var rc = ParseReferentialAction();
            if (rc.HasError)
            {
                return rc.Error;
            }

            onDelete = rc.Result;
        }

        var onUpdate = ReferentialAction.NoAction;

        if (TryKeywords("ON", "UPDATE"))
        {
            var rc = ParseReferentialAction();
            if (rc.HasError)
            {
                return rc.Error;
            }

            onUpdate = rc.Result;
        }

        var notForReplication = TryKeywords("NOT", "FOR", "REPLICATION");
        return CreateParseResult(new SqlConstraintForeignKey
        {
            Columns = columns,
            ReferencedTableName = tableName.Word,
            RefColumn = refColumn,
            OnDeleteAction = onDelete,
            OnUpdateAction = onUpdate,
            NotForReplication = notForReplication,
        });
    }

    public ParseResult<SqlTopClause> Parse_TopClause()
    {
        if (!TryKeyword("TOP"))
        {
            return NoneResult<SqlTopClause>();
        }

        var expression = ParseValue();
        if (expression.HasError)
        {
            return expression.Error;
        }

        if (expression.Result == null)
        {
            return CreateParseError("Expected TOP expression");
        }

        var topClause = new SqlTopClause()
        {
            Expression = expression.ResultValue
        };
        if (TryKeyword("PERCENT"))
        {
            topClause.IsPercent = true;
        }

        if (TryKeywords("WITH", "TIES"))
        {
            topClause.IsWithTies = true;
        }

        return topClause;
    }

    public ParseResult<SelectStatement> ParseSelectStatement()
    {
        if (!TryKeyword("SELECT"))
        {
            return NoneResult<SelectStatement>();
        }

        var selectStatement = new SelectStatement();

        var selectTypeClause = Parse_SelectTypeClause();
        if (selectTypeClause.HasError)
        {
            return selectTypeClause.Error;
        }

        selectStatement.SelectType = selectTypeClause.ResultValue;

        var topClause = Parse_TopClause();
        if (topClause.HasError)
        {
            return topClause.Error;
        }

        if (topClause.Result != null)
        {
            selectStatement.Top = topClause.Result;
        }

        var columns = Parse_SelectColumns();
        if (columns.HasError)
        {
            return columns.Error;
        }

        selectStatement.Columns = columns.ResultValue;

        if (TryKeyword("FROM"))
        {
            var tableName = ReadSqlIdentifier().Word;
            var tableSource = new SqlTableSource()
            {
                TableName = tableName
            };
            if (TryKeyword("WITH"))
            {
                MatchString("(");
                var tableHints = ParseWithComma<SqlHint>(() =>
                {
                    var hint = ReadSqlIdentifier().Word;
                    return new SqlHint()
                    {
                        Name = hint
                    };
                });
                if (tableHints.HasError)
                {
                    return tableHints.Error;
                }

                MatchString(")");
                tableSource.Withs = tableHints.ResultValue;
            }

            selectStatement.From = tableSource;
        }

        if (TryKeyword("WHERE"))
        {
            var rc = Parse_WhereExpression();
            if (rc.HasError)
            {
                return rc.Error;
            }

            selectStatement.Where = rc.Result;
        }

        var orderByClause = ParseOrderByClause();
        if (orderByClause.HasError)
        {
            return orderByClause.Error;
        }

        selectStatement.OrderBy = orderByClause.Result;

        SkipStatementEnd();
        return CreateParseResult(selectStatement);
    }

    private ParseResult<SelectColumn> Parse_Column_Star()
    {
        if (TryMatch("*"))
        {
            return new SelectColumn
            {
                ColumnName = "*"
            };
        }

        return NoneResult<SelectColumn>();
    }
    
    private ParseResult<SelectColumn> Parse_Column_Identifier()
    {
        if (TryReadSqlIdentifier(out var fieldName))
        {
            return new SelectColumn()
            {
                ColumnName = fieldName.Word
            };
        }
        return NoneResult<SelectColumn>();
    }
    
    private ParseResult<SelectSubQueryColumn> Parse_Column_SubQuery()
    {
        if (Try(ParseValue, out var valueExpr))
        {
            return new SelectSubQueryColumn
            {
                SubQuery = valueExpr.ResultValue,
            };
        }
        return NoneResult<SelectSubQueryColumn>();
    }

    private ParseResult<List<ISelectColumnExpression>> Parse_SelectColumns()
    {
        var columns = ParseWithComma<ISelectColumnExpression>(() =>
        {
            if (IsPeekKeywords("FROM"))
            {
                return NoneResult<ISelectColumnExpression>();
            }
            
            var column = Or<ISelectColumnExpression>(
                Parse_Column_Star, 
                Parse_Column_Identifier, 
                Parse_Column_SubQuery)();
            if (column.HasError)
            {
                return column.Error;
            }

            if (TryKeyword("AS"))
            {
                var aliasName = ParseSqlIdentifier();
                if(aliasName.HasError)
                {
                    return aliasName.Error;
                }
                column.ResultValue.Alias = aliasName.ResultValue.Value;
            }

            if (TryMatch("="))
            {
                var rightExpr = ParseArithmeticExpr();
                if (rightExpr.HasError)
                {
                    return rightExpr.Error;
                }

                return new SelectSubQueryColumn
                {
                    SubQuery = new SqlAssignExpr()
                    { 
                        Left = column.ResultValue,
                        Right = rightExpr.ResultValue,
                    },
                    Alias = column.ResultValue.Alias,
                };
            }
            
            return column;
        });
        return columns;
    }

    private ParseResult<SqlOrderByClause> ParseOrderByClause()
    {
        if (!TryKeywords("ORDER", "BY"))
        {
            return NoneResult<SqlOrderByClause>();
        }

        var orderByColumns = ParseWithComma<SqlOrderByColumn>(() =>
        {
            var column = ReadSqlIdentifier().Word;
            var order = OrderType.Asc;
            if (TryKeyword("ASC"))
            {
                order = OrderType.Asc;
            }
            else if (TryKeyword("DESC"))
            {
                order = OrderType.Desc;
            }

            return new SqlOrderByColumn
            {
                ColumnName = column,
                Order = order
            };
        });
        if (orderByColumns.HasError)
        {
            return orderByColumns.Error;
        }

        return new SqlOrderByClause
        {
            Columns = orderByColumns.ResultValue
        };
    }

    private ParseResult<ISqlExpression> Parse_WhereExpression()
    {
        var rc = Or<ISqlExpression>(Parse_SearchCondition, Parse_ConditionExpression)();
        if (rc.HasError)
        {
            return rc.Error;
        }

        if (Try(Parse_LogicalOperator, out var logicalOperator))
        {
            var rightExprResult = Parse_WhereExpression();
            if (rightExprResult.HasError)
            {
                return rightExprResult.Error;
            }

            return new SqlSearchCondition
            {
                Left = rc.ResultValue,
                LogicalOperator = logicalOperator.Result!.Value,
                Right = rightExprResult.Result
            };
        }

        return rc;
    }

    private ParseResult<SqlSearchCondition> Parse_SearchCondition()
    {
        var startPosition = _text.Position;
        var leftExpr = ParseValue();
        if (leftExpr.HasError)
        {
            return leftExpr.Error;
        }

        if (leftExpr.Result == null)
        {
            return CreateParseError("Expected left search expression");
        }

        var logicalOperator = Parse_LogicalOperator();
        if (logicalOperator.HasError)
        {
            return logicalOperator.Error;
        }

        if (logicalOperator.Result == null)
        {
            _text.Position = startPosition;
            return NoneResult<SqlSearchCondition>();
        }

        var rightExpr = ParseValue();
        if (rightExpr.HasError)
        {
            return rightExpr.Error;
        }

        if (rightExpr.Result == null)
        {
            return CreateParseError("Expected right search expression");
        }

        return new SqlSearchCondition()
        {
            Left = leftExpr.ResultValue,
            LogicalOperator = logicalOperator.Result.Value,
            Right = rightExpr.ResultValue
        };
    }

    private ParseResult<SqlConditionExpression> Parse_ConditionExpression()
    {
        var startPosition = _text.Position;
        var leftExpr = ParseValue();
        if (leftExpr.HasError)
        {
            return leftExpr.Error;
        }

        if (leftExpr.Result == null)
        {
            return CreateParseError("Expected left expression");
        }

        var operation = Parse_ComparisonOperator();
        if (operation.HasError)
        {
            return operation.Error;
        }

        if (operation.Result == null)
        {
            _text.Position = startPosition;
            return NoneResult<SqlConditionExpression>();
        }

        var rightExpr = ParseArithmeticExpr();
        if (rightExpr.HasError)
        {
            return rightExpr.Error;
        }

        if (rightExpr.Result == null)
        {
            return CreateParseError("Expected right expression");
        }

        return new SqlConditionExpression()
        {
            Left = leftExpr.ResultValue,
            ComparisonOperator = operation.Result.Value,
            Right = rightExpr.ResultValue
        };
    }

    private ParseResult<ComparisonOperator?> Parse_ComparisonOperator()
    {
        var rc = Or(
            Keywords("IS", "NOT"),
            Keywords("LIKE"),
            Keywords("IN"),
            Symbol("<>"),
            Symbol(">="),
            Symbol("<="),
            Symbol("="),
            Symbol(">"),
            Symbol("<")
        )();
        if (rc.HasError)
        {
            return rc.Error;
        }

        if (rc.Result == null)
        {
            return NoneResult<ComparisonOperator?>();
        }

        var comparisonOperator = rc.Result.Value.ToUpper() switch
        {
            "IS NOT" => ComparisonOperator.IsNot,
            "LIKE" => ComparisonOperator.Like,
            "IN" => ComparisonOperator.In,
            "<>" => ComparisonOperator.NotEqual,
            ">=" => ComparisonOperator.GreaterThanOrEqual,
            "<=" => ComparisonOperator.LessThanOrEqual,
            "=" => ComparisonOperator.Equal,
            ">" => ComparisonOperator.GreaterThan,
            "<" => ComparisonOperator.LessThan,
            _ => ComparisonOperator.Equal
        };
        return comparisonOperator;
    }

    private ParseResult<LogicalOperator?> Parse_LogicalOperator()
    {
        var rc = Or(Keywords("AND"), Keywords("OR"), Keywords("NOT"))();
        if (rc.HasError)
        {
            return rc.Error;
        }

        if (rc.Result == null)
        {
            return NoneResult<LogicalOperator?>();
        }

        var logicalOperator = rc.Result.Value.ToUpper() switch
        {
            "AND" => LogicalOperator.And,
            "OR" => LogicalOperator.Or,
            "NOT" => LogicalOperator.Not,
            _ => LogicalOperator.None
        };
        return logicalOperator;
    }

    public void SkipStatementEnd()
    {
        var ch = _text.PeekChar();
        if (ch == ';')
        {
            _text.ReadChar();
        }
    }

    public bool Try<T>(Func<ParseResult<T>> parseFunc, out ParseResult<T> result)
    {
        var localResult = parseFunc();
        if (localResult.HasError)
        {
            result = localResult;
            return true;
        }

        if (localResult.Result == null)
        {
            result = localResult;
            return false;
        }

        result = localResult;
        return true;
    }

    private ParseError CreateParseError(string error)
    {
        return new ParseError(error)
        {
            Offset = _text.Position
        };
    }

    private ParseResult<T> CreateParseResult<T>(T result)
    {
        return new ParseResult<T>(result);
    }

    private T? GetResult<T>(Func<ParseResult<T>> parseFn)
    {
        var result = parseFn();
        return result.Result;
    }

    private bool IsAny<T>(params Func<ParseResult<T>>[] parseFnList)
    {
        var span = Or(parseFnList)();
        if (span.HasError)
        {
            return false;
        }

        return span.Result != null;
    }

    private bool IsPeekKeywords(params string[] keywords)
    {
        var keywordsResult = PeekKeywords(keywords)();
        if (keywordsResult.Result != null && keywordsResult.Result.Value.Length != 0)
        {
            return true;
        }

        return false;
    }

    private Func<ParseResult<SqlToken>> Keywords(params string[] keywords)
    {
        return () => ParseKeywords(keywords);
    }

    private Func<ParseResult<SqlToken>> Symbol(string symbol)
    {
        return () =>
        {
            SkipWhiteSpace();
            if (_text.TryMatch(symbol))
            {
                return new SqlToken
                {
                    Value = symbol
                };
            }

            return NoneResult<SqlToken>();
        };
    }


    private void MatchString(string expected)
    {
        SkipWhiteSpace();
        _text.MatchSymbol(expected);
    }

    private ParseResult<T> NoneResult<T>()
    {
        return new ParseResult<T>(default(T));
    }

    private Func<ParseResult<T>> One<T>(params Func<ParseResult<T>>[] parseFnList)
    {
        return () =>
        {
            var rc = Or(parseFnList)();
            if (rc.HasError)
            {
                return rc;
            }

            if (rc.Result != null)
            {
                return rc;
            }

            return CreateParseError("Expected one of the options");
        };
    }

    private Func<ParseResult<T>> Or<T>(params Func<IParseResult>[] parseFnList)
    {
        return () =>
        {
            foreach (var parseFn in parseFnList)
            {
                var rc = parseFn();
                if (rc.HasError)
                {
                    return rc.Error;
                }

                if (rc is { HasResult: true, Object: not null })
                {
                    return CreateParseResult((T)rc.ObjectValue);
                }
            }

            return NoneResult<T>();
        };
    }


    private Func<ParseResult<T>> Or<T>(params Func<ParseResult<T>>[] parseFnList)
    {
        return () =>
        {
            foreach (var parseFn in parseFnList)
            {
                var rc = parseFn();
                if (rc.HasError)
                {
                    return rc.Error;
                }

                if (rc is { HasResult: true, Object: not null })
                {
                    return rc.ResultValue;
                }
            }

            return NoneResult<T>();
        };
    }

    private ParseResult<SqlColumnDefinition> ParseColumnConstraints(SqlColumnDefinition sqlColumn)
    {
        if (IsPeekKeywords(","))
        {
            return NoneResult<SqlColumnDefinition>();
        }

        do
        {
            var startPosition = _text.Position;
            if (TryKeywords("PRIMARY", "KEY"))
            {
                // 最後一個column 有可能沒有逗號 又寫 Table Constraint 的話會被誤判, 所以要檢查是否有 CLUSTERED 
                if (TryKeyword("CLUSTERED"))
                {
                    _text.Position = startPosition;
                    break;
                }

                sqlColumn.IsPrimaryKey = true;
                continue;
            }

            if (Try(ParseIdentity, out var identityResult))
            {
                if (identityResult.HasError)
                {
                    return identityResult.Error;
                }

                sqlColumn.Identity = identityResult.ResultValue;
                continue;
            }

            if (Try(ParseDefaultValue, out var defaultValue))
            {
                if (identityResult.HasError)
                {
                    return identityResult.Error;
                }

                sqlColumn.Constraints.Add(defaultValue.ResultValue);
                continue;
            }

            var constraintStartPosition = _text.Position;
            if (TryMatch(ConstraintKeyword))
            {
                var constraintName = _text.ReadSqlIdentifier();
                if (Try(ParseDefaultValue, out var constraintDefaultValue))
                {
                    if (identityResult.HasError)
                    {
                        return identityResult.Error;
                    }

                    var subConstraint = constraintDefaultValue.ResultValue;
                    subConstraint.ConstraintName = constraintName.Word;
                    sqlColumn.Constraints.Add(subConstraint);
                    continue;
                }

                _text.Position = constraintStartPosition;
                var columnConstraint = ParseTableConstraint();
                if (columnConstraint.HasError)
                {
                    return columnConstraint.Error;
                }

                if (columnConstraint.Result == null)
                {
                    return CreateParseError("Expect Constraint DEFAULT");
                }

                sqlColumn.Constraints.Add(columnConstraint.Result);
            }

            if (TryKeywords("NOT", "FOR", "REPLICATION"))
            {
                sqlColumn.NotForReplication = true;
                continue;
            }

            if (TryKeywords("NOT", "NULL"))
            {
                sqlColumn.IsNullable = false;
                continue;
            }

            if (TryKeywords("NULL"))
            {
                sqlColumn.IsNullable = true;
                continue;
            }

            break;
        } while (true);

        return CreateParseResult(sqlColumn);
    }

    private ParseResult<SqlColumnDefinition> ParseColumnDefinition()
    {
        var startPosition = _text.Position;
        if (!TryReadSqlIdentifier(out var columnNameSpan))
        {
            return NoneResult<SqlColumnDefinition>();
        }

        var columnDefinition = ParseColumnTypeDefinition(columnNameSpan);
        if (columnDefinition.HasError)
        {
            return columnDefinition.Error;
        }

        if (columnDefinition.Result == null)
        {
            _text.Position = startPosition;
            return NoneResult<SqlColumnDefinition>();
        }

        var c = ParseColumnConstraints(columnDefinition.Result);
        if (c.HasError)
        {
            return c.Error;
        }

        return CreateParseResult(columnDefinition.ResultValue);
    }

    private ParseResult<List<SqlConstraintColumn>> ParseColumnsAscDesc()
    {
        var columns = ParseParenthesesWithComma(() =>
        {
            var columnName = _text.ReadSqlIdentifier();
            var order = string.Empty;
            if (TryKeyword("ASC"))
            {
                order = "ASC";
            }
            else if (TryKeyword("DESC"))
            {
                order = "DESC";
            }

            return CreateParseResult(new SqlConstraintColumn
            {
                ColumnName = columnName.Word,
                Order = order,
            });
        });
        return columns;
    }

    private ParseResult<SqlDataSize> Parse_DataSize()
    {
        if (!TryMatch("("))
        {
            return NoneResult<SqlDataSize>();
        }

        var dataSize = new SqlDataSize();
        if (_text.TryKeywordIgnoreCase("MAX"))
        {
            dataSize.Size = "MAX";
            _text.MatchSymbol(")");
            return dataSize;
        }

        dataSize.Size = _text.ReadInt().Word;
        if (_text.PeekChar() == ',')
        {
            _text.ReadChar();
            dataSize.Scale = int.Parse(_text.ReadInt().Word);
        }

        if (!TryMatch(")"))
        {
            return CreateParseError("Expected )");
        }

        return dataSize;
    }

    private ParseResult<SqlDataType> Parse_DataType()
    {
        var startPosition = _text.Position;
        if (!TryReadSqlIdentifier(out var identifier))
        {
            return NoneResult<SqlDataType>();
        }

        var dataType = Parse_DataSize();
        if (dataType.HasError)
        {
            _text.Position = startPosition;
            return dataType.Error;
        }

        return new SqlDataType()
        {
            DataTypeName = identifier.Word,
            Size = dataType.Result != null ? dataType.ResultValue : new SqlDataSize()
        };
    }

    private ParseResult<SqlColumnDefinition> ParseColumnTypeDefinition(TextSpan columnNameSpan)
    {
        var column = new SqlColumnDefinition
        {
            ColumnName = columnNameSpan.Word,
            DataType = ReadSqlIdentifier().Word
        };

        var dataSize = Parse_DataSize();
        if (dataSize.HasError)
        {
            return dataSize.Error;
        }

        column.DataSize = dataSize.Result;
        return CreateParseResult(column);
    }

    /*
     * <computed_column_definition> ::=
column_name AS computed_column_expression
[ PERSISTED [ NOT NULL ] ]
[
    [ CONSTRAINT constraint_name ]
    { PRIMARY KEY | UNIQUE }
        [ CLUSTERED | NONCLUSTERED ]
        [
            WITH FILLFACTOR = fillfactor
          | WITH ( <index_option> [ ,... n ] )
        ]
        [ ON { partition_scheme_name ( partition_column_name )
        | filegroup | "default" } ]

    | [ FOREIGN KEY ]
        REFERENCES referenced_table_name [ ( ref_column ) ]
        [ ON DELETE { NO ACTION | CASCADE } ]
        [ ON UPDATE { NO ACTION } ]
        [ NOT FOR REPLICATION ]

    | CHECK [ NOT FOR REPLICATION ] ( logical_expression )
]
     */
    private ParseResult<SqlComputedColumnDefinition> ParseComputedColumnDefinition()
    {
        var startPosition = _text.Position;
        if (!TryReadSqlIdentifier(out var columnNameSpan))
        {
            return NoneResult<SqlComputedColumnDefinition>();
        }

        if (!TryKeyword("AS"))
        {
            _text.Position = startPosition;
            return NoneResult<SqlComputedColumnDefinition>();
        }

        if (!TryMatch("("))
        {
            _text.Position = startPosition;
            return CreateParseError("Expected (");
        }

        var computedColumnExpressionSpan = _text.ReadUntilRightParenthesis();

        if (!TryMatch(")"))
        {
            _text.Position = startPosition;
            return CreateParseError("Expected )");
        }

        var persist = TryKeyword("PERSISTED");
        var notNull = TryKeywords("NOT", "NULL");

        return CreateParseResult(new SqlComputedColumnDefinition
        {
            ColumnName = columnNameSpan.Word,
            Expression = computedColumnExpressionSpan.Word,
            IsPersisted = persist,
            IsNotNull = notNull
        });
    }

    private ParseResult<SqlConstraintDefaultValue> ParseDefaultValue()
    {
        if (!TryKeyword("DEFAULT"))
        {
            return NoneResult<SqlConstraintDefaultValue>();
        }

        TextSpan defaultValue;
        if (TryMatch("("))
        {
            defaultValue = _text.ReadUntilRightParenthesis();
            _text.MatchSymbol(")");
            return CreateParseResult(new SqlConstraintDefaultValue
            {
                ConstraintName = string.Empty,
                DefaultValue = defaultValue.Word
            });
        }

        var nullValue = _text.PeekIdentifier("NULL");
        if (nullValue.Length > 0)
        {
            _text.ReadIdentifier();
            return CreateParseResult(new SqlConstraintDefaultValue
            {
                ConstraintName = string.Empty,
                DefaultValue = nullValue.Word
            });
        }

        if (_text.Try(_text.ReadSqlIdentifier, out var funcName))
        {
            _text.MatchSymbol("(");
            var funcArgs = _text.ReadUntilRightParenthesis();
            _text.MatchSymbol(")");
            defaultValue = new TextSpan
            {
                Word = $"{funcName.Word}({funcArgs.Word})",
                Offset = funcName.Offset,
                Length = funcName.Length + funcArgs.Length + 2
            };
            return CreateParseResult(new SqlConstraintDefaultValue
            {
                ConstraintName = string.Empty,
                DefaultValue = defaultValue.Word,
            });
        }

        if (_text.Try(_text.ReadSqlQuotedString, out var quotedString))
        {
            return CreateParseResult(new SqlConstraintDefaultValue
            {
                ConstraintName = string.Empty,
                DefaultValue = quotedString.Word,
            });
        }

        if (_text.Try(_text.ReadSqlDate, out var date))
        {
            return CreateParseResult(new SqlConstraintDefaultValue
            {
                ConstraintName = string.Empty,
                DefaultValue = date.Word,
            });
        }

        if (_text.Try(_text.ReadNegativeNumber, out var negativeNumber))
        {
            return CreateParseResult(new SqlConstraintDefaultValue
            {
                ConstraintName = string.Empty,
                DefaultValue = negativeNumber.Word,
            });
        }

        if (_text.Try(_text.ReadFloat, out var floatNumber))
        {
            return CreateParseResult(new SqlConstraintDefaultValue
            {
                ConstraintName = string.Empty,
                DefaultValue = floatNumber.Word,
            });
        }

        defaultValue = _text.ReadInt();
        return CreateParseResult(new SqlConstraintDefaultValue
        {
            ConstraintName = string.Empty,
            DefaultValue = defaultValue.Word,
        });
    }

    private ParseResult<SqlIdentity> ParseIdentity()
    {
        if (!TryMatch("IDENTITY"))
        {
            return NoneResult<SqlIdentity>();
        }

        var sqlIdentity = new SqlIdentity
        {
            Seed = 1,
            Increment = 1
        };
        if (TryMatch("("))
        {
            sqlIdentity.Seed = long.Parse(_text.ReadInt().Word);
            _text.MatchSymbol(",");
            sqlIdentity.Increment = int.Parse(_text.ReadInt().Word);
            _text.MatchSymbol(")");
        }

        return CreateParseResult(sqlIdentity);
    }

    private ParseResult<SqlValue> ParseNumberValue()
    {
        var startPosition = _text.Position;
        var negative = TryMatch("-");
        var number = Or(ParseFloatValue, ParseIntValue)();
        if (number.HasError)
        {
            _text.Position = startPosition;
            return number.Error;
        }
        if(number.Result == null)
        {
            _text.Position = startPosition;
            return NoneResult<SqlValue>();
        }
        number.ResultValue.Value = negative ? $"-{number.ResultValue.Value}" : number.ResultValue.Value;
        return number;
    }

    private ParseResult<SqlValue> ParseFloatValue()
    {
        if (_text.Try(_text.ReadFloat, out var floatNumber))
        {
            return new SqlValue
            {
                Value = floatNumber.Word
            };
        }
        return NoneResult<SqlValue>();
    }

    private ParseResult<SqlValue> ParseIntValue()
    {
        if (_text.Try(_text.ReadInt, out var number))
        {
            return CreateParseResult(new SqlValue
            {
                SqlType = SqlType.IntValue,
                Value = number.Word
            });
        }

        return NoneResult<SqlValue>();
    }

    public ParseResult<SelectType> Parse_SelectTypeClause()
    {
        var rc = Or(Keywords("ALL"), Keywords("DISTINCT"))();
        if (rc.HasError)
        {
            return rc.Error;
        }

        if (rc is not { HasValue: true, Result: not null })
        {
            return SelectType.All;
        }

        var selectType = rc.Result.Value.ToUpper() switch
        {
            "ALL" => SelectType.All,
            "DISTINCT" => SelectType.Distinct,
            _ => SelectType.All
        };
        return selectType;
    }

    private ParseResult<SqlToken> ParseKeywords(params string[] keywords)
    {
        if (TryKeywords(keywords))
        {
            return CreateParseResult(new SqlToken
            {
                Value = string.Join(" ", keywords)
            });
        }

        return NoneResult<SqlToken>();
    }

    private ParseResult<SqlParameterValue> ParseParameterAssignValue()
    {
        SkipWhiteSpace();
        if (!_text.Try(_text.ReadSqlIdentifier, out var name))
        {
            return NoneResult<SqlParameterValue>();
        }

        if (!_text.TryMatch("="))
        {
            return CreateParseError("Expected =");
        }

        if (!_text.Try(_text.ReadSqlQuotedString, out var nameValue))
        {
            return CreateParseError($"Expected @name value, but got {_text.PreviousWord().Word}");
        }

        return CreateParseResult(new SqlParameterValue
        {
            Name = name.Word,
            Value = nameValue.Word
        });
    }

    private ParseResult<SqlParameterValue> ParseParameterValue()
    {
        SkipWhiteSpace();
        var startPosition = _text.Position;
        var valueResult = ParseValue();
        if (valueResult.HasError)
        {
            _text.Position = startPosition;
            return valueResult.Error;
        }

        if (valueResult.Result == null)
        {
            return NoneResult<SqlParameterValue>();
        }

        if (_text.Peek(_text.ReadSymbols).Word == "=")
        {
            _text.Position = startPosition;
            return NoneResult<SqlParameterValue>();
        }

        return new SqlParameterValue
        {
            Name = string.Empty,
            Value = valueResult.ResultValue.Value
        };
    }

    private ParseResult<SqlParameterValue> ParseParameterValueOrAssignValue()
    {
        var rc1 = ParseParameterValue();
        if (rc1.HasError)
        {
            return rc1.Error;
        }

        if (rc1.HasResult && rc1.Result != null)
        {
            return rc1;
        }

        var rc2 = ParseParameterAssignValue();
        if (rc2.HasError)
        {
            return rc2.Error;
        }

        if (rc2.HasResult && rc2.Result != null)
        {
            return rc2;
        }

        return NoneResult<SqlParameterValue>();
    }

    private ParseResult<List<T>> ParseParenthesesWithComma<T>(Func<ParseResult<T>> parseElemFn)
    {
        if (!TryMatch("("))
        {
            return CreateParseError("Expected (");
        }

        var elements = ParseWithComma(parseElemFn);
        if (elements.HasError)
        {
            return elements.Error;
        }

        if (!TryMatch(")"))
        {
            return CreateParseError("Expected )");
        }

        return elements;
    }

    private ParseResult<SqlConstraintPrimaryKeyOrUnique> ParsePrimaryKeyOrUnique()
    {
        var sqlConstraint = new SqlConstraintPrimaryKeyOrUnique();
        var primaryKeyOrUniqueToken = GetResult(Or(Keywords("PRIMARY", "KEY"), Keywords("UNIQUE")));
        if (primaryKeyOrUniqueToken != null)
        {
            sqlConstraint.ConstraintType = primaryKeyOrUniqueToken.Value;
        }

        if (string.IsNullOrEmpty(sqlConstraint.ConstraintType))
        {
            return NoneResult<SqlConstraintPrimaryKeyOrUnique>();
        }

        var clusteredToken = GetResult(Or(Keywords("CLUSTERED"), Keywords("NONCLUSTERED")));
        if (clusteredToken != null)
        {
            sqlConstraint.Clustered = clusteredToken.Value;
        }

        var columnsSpan = ParseColumnsAscDesc();
        if (columnsSpan.HasError)
        {
            return columnsSpan.Error;
        }

        sqlConstraint.Columns = columnsSpan.ResultValue;
        return sqlConstraint;
    }

    private ParseResult<SqlConstraintPrimaryKeyOrUnique> ParsePrimaryKeyOrUniqueExpression()
    {
        var primaryKeyOrUniqueSpan = ParsePrimaryKeyOrUnique();
        if (primaryKeyOrUniqueSpan.HasError)
        {
            return primaryKeyOrUniqueSpan.Error;
        }

        if (primaryKeyOrUniqueSpan.Result == null)
        {
            return NoneResult<SqlConstraintPrimaryKeyOrUnique>();
        }

        var sqlConstraint = primaryKeyOrUniqueSpan.ResultValue;
        if (TryKeyword("WITH"))
        {
            var togglesSpan = ParseParenthesesWithComma(ParseWithToggle);
            if (togglesSpan.HasError)
            {
                return togglesSpan.Error;
            }

            sqlConstraint.WithToggles = togglesSpan.ResultValue;
        }

        if (TryKeyword("ON"))
        {
            sqlConstraint.On = ReadSqlIdentifier().Word;
        }

        if (Try(ParseIdentity, out var identitySpan))
        {
            if (identitySpan.HasError)
            {
                return identitySpan.Error;
            }

            sqlConstraint.Identity = identitySpan.ResultValue;
        }

        return sqlConstraint;
    }

    private ParseResult<ReferentialAction> ParseReferentialAction()
    {
        var actionTokenSpan = One(Keywords("NO", "ACTION"), Keywords("CASCADE"), Keywords("SET", "NULL"),
            Keywords("SET", "DEFAULT"))();
        if (actionTokenSpan.HasError)
        {
            return actionTokenSpan.Error;
        }

        var actionToken = actionTokenSpan.ResultValue;
        var action = actionToken.Value.ToUpper() switch
        {
            "NO ACTION" => ReferentialAction.NoAction,
            "CASCADE" => ReferentialAction.Cascade,
            "SET NULL" => ReferentialAction.SetNull,
            "SET DEFAULT" => ReferentialAction.SetDefault,
            _ => ReferentialAction.NoAction
        };
        return new ParseResult<ReferentialAction>(action);
    }

    private ParseResult<ISqlConstraint> ParseTableConstraint()
    {
        var constraintName = string.Empty;
        if (TryKeyword(ConstraintKeyword))
        {
            constraintName = ReadSqlIdentifier().Word;
        }

        var tablePrimaryKeyOrUniqueExpr = ParsePrimaryKeyOrUniqueExpression();
        if (tablePrimaryKeyOrUniqueExpr.HasError)
        {
            return tablePrimaryKeyOrUniqueExpr.Error;
        }

        if (tablePrimaryKeyOrUniqueExpr.Result != null)
        {
            tablePrimaryKeyOrUniqueExpr.Result.ConstraintName = constraintName;
            return tablePrimaryKeyOrUniqueExpr.Result;
        }

        var tableForeignKeyExpr = ParseForeignKeyExpression();
        if (tableForeignKeyExpr.HasError)
        {
            return tableForeignKeyExpr.Error;
        }

        if (tableForeignKeyExpr.Result != null)
        {
            tableForeignKeyExpr.Result.ConstraintName = constraintName;
            return tableForeignKeyExpr.Result;
        }

        return NoneResult<ISqlConstraint>();
    }

    private ParseResult<SqlFieldExpression> ParseTableName()
    {
        if (_text.Try(_text.ReadIdentifier, out var fieldName))
        {
            return CreateParseResult(new SqlFieldExpression()
            {
                FieldName = fieldName.Word
            });
        }

        return NoneResult<SqlFieldExpression>();
    }

    private ParseResult<SqlValues> ParseValues()
    {
        var startPosition = _text.Position;
        if(!TryMatch("("))
        {
            return NoneResult<SqlValues>();
        }
        var items = ParseWithComma(() =>
        {
            var value = ParseValue();
            if (value.HasError)
            {
                return value.Error;
            }
            return value;
        });
        if (items.HasError)
        {
            _text.Position = startPosition;
            return items.Error;
        }
        if (!TryMatch(")"))
        {
            return CreateParseError("Expected )");
        }
        if(items.ResultValue.Count == 1)
        {
            _text.Position = startPosition;
            return NoneResult<SqlValues>();
        }
        return new SqlValues
        {
            Items = items.ResultValue.ToList()
        };
    }

    public ParseResult<ISqlValue> ParseValue()
    {
        if(Try(ParseValues, out var values))
        {
            return values.ResultValue;
        }
        
        if (TryMatch("("))
        {
            var value = ParseValue();
            if (value.HasError)
            {
                return value.Error;
            }
            MatchString(")");
            return new SqlGroup
            {
                Inner = value.ResultValue
            };
        }

        if (TryKeyword("NULL"))
        {
            return new SqlNullValue();
        }
        
        if(Try(ParseNumberValue, out var numberValue))
        {
            return numberValue.ResultValue;
        }

        if (_text.Try(_text.ReadSqlQuotedString, out var quotedString))
        {
            return new SqlValue
            {
                Value = quotedString.Word
            };
        }

        if (Try(Parse_FunctionName, out var function))
        {
            return function.ResultValue;
        }

        if (TryReadSqlIdentifier(out var identifier))
        {
            if (TryKeyword("AS"))
            {
                var dataType = Parse_DataType();
                return new SqlAsExpr
                {
                    Instance = new SqlFieldExpression
                    {
                        FieldName = identifier.Word
                    },
                    DataType = dataType.ResultValue
                };
            }

            return new SqlFieldExpression
            {
                FieldName = identifier.Word
            };
        }

        if (Try(ParseTableName, out var tableName))
        {
            return tableName.ResultValue;
        }

        return NoneResult<ISqlValue>();
    }

    private ParseResult<SqlFunctionExpression> Parse_FunctionName()
    {
        var startPosition = _text.Position;
        if (TryReadSqlIdentifier(out var identifier))
        {
            if (TryMatch("("))
            {
                var parameters = ParseWithComma(ParseValue);
                MatchString(")");
                return new SqlFunctionExpression
                {
                    FunctionName = identifier.Word,
                    Parameters = parameters.ResultValue!.ToArray()
                };
            }
        }

        _text.Position = startPosition;
        return NoneResult<SqlFunctionExpression>();
    }

    private ParseResult<List<T>> ParseWithComma<T>(Func<ParseResult<T>> parseElemFn)
    {
        var elements = new List<T>();
        do
        {
            var elem = parseElemFn();
            if (elem is { HasResult: true, Result: null })
            {
                break;
            }

            if (elem.HasError)
            {
                return elem.Error;
            }

            elements.Add(elem.ResultValue);
            if (_text.PeekChar() != ',')
            {
                break;
            }

            _text.ReadChar();
        } while (!_text.IsEnd());

        return CreateParseResult(elements);
    }

    private ParseResult<SqlToggle> ParseWithToggle()
    {
        var startPosition = _text.Position;
        var toggleName = _text.ReadSqlIdentifier();
        if (toggleName.Length == 0)
        {
            _text.Position = startPosition;
            return NoneResult<SqlToggle>();
        }

        var toggle = new SqlToggle
        {
            ToggleName = toggleName.Word
        };

        if (!_text.TryMatch("="))
        {
            _text.Position = startPosition;
            return CreateParseError("Expected toggleName =");
        }

        if (_text.Try(_text.ReadInt, out var number))
        {
            toggle.Value = number.Word;
            return CreateParseResult(toggle);
        }

        toggle.Value = _text.ReadSqlIdentifier().Word;
        return CreateParseResult(toggle);
    }

    private Func<ParseResult<SqlToken>> PeekKeywords(params string[] keywords)
    {
        return () =>
        {
            var startPosition = _text.Position;
            var result = ParseKeywords(keywords);
            _text.Position = startPosition;
            return result;
        };
    }
    
    private string PeekSymbolString(int length)
    {
        SkipWhiteSpace();
        var startPosition = _text.Position;
        var span = _text.ReadString(length);
        _text.Position = startPosition;
        return span.Word;
    }

    private Func<ParseResult<SqlToken>> PeekSymbol(int length)
    {
        return () =>
        {
            SkipWhiteSpace();
            var startPosition = _text.Position;
            var span = _text.ReadString(length);
            var token = new SqlToken
            {
                Value = span.Word
            };
            _text.Position = startPosition;
            return token;
        };
    }

    private string ReadSymbolString(int length)
    {
        SkipWhiteSpace();
        var span = _text.ReadString(length);
        return span.Word;
    }

    private TextSpan ReadSqlIdentifier()
    {
        SkipWhiteSpace();
        return _text.ReadSqlIdentifier();
    }

    private void SkipWhiteSpace()
    {
        while (true)
        {
            var isSkip1 = _text.SkipWhitespace();
            var isSkip2 = _text.SkipSqlComment();
            var isSkip = isSkip1 || isSkip2;
            if (!isSkip)
            {
                break;
            }
        }
    }

    private bool TryMatch(string expected)
    {
        SkipWhiteSpace();
        return _text.TryMatch(expected);
    }

    private Func<ParseResult<SqlToken>> ParseSymbol(string expected)
    {
        return () =>
        {
            if(TryMatch(expected))
            {
                return new SqlToken
                {
                    Value = expected
                };
            }
            return NoneResult<SqlToken>();
        };
    }

    private bool TryKeyword(string expected)
    {
        SkipWhiteSpace();
        return _text.TryKeywordIgnoreCase(expected);
    }

    private bool TryKeywords(params string[] keywords)
    {
        SkipWhiteSpace();
        return _text.TryKeywordsIgnoreCase(keywords);
    }

    private bool TryPeekKeyword(string expected)
    {
        SkipWhiteSpace();
        var tmpPosition = _text.Position;
        var isSuccess = _text.TryKeywordIgnoreCase(expected);
        _text.Position = tmpPosition;
        return isSuccess;
    }

    private bool TryReadSqlIdentifier(out TextSpan result)
    {
        SkipWhiteSpace();
        return _text.Try(_text.ReadSqlIdentifier, out result);
    }

    private ParseResult<SqlToken> ParseSqlIdentifier()
    {
        if(TryReadSqlIdentifier(out var identifier))
        {
            return new SqlToken
            {
                Value = identifier.Word
            };
        }
        return NoneResult<SqlToken>();
    }

    private bool TryReadInt(out TextSpan result)
    {
        SkipWhiteSpace();
        return _text.Try(_text.ReadInt, out result);
    }

    public ParseResult<ISqlExpression> ParseArithmeticExpr()
    {
        return ParseArithmetic_Step1_AdditionOrSubtraction();
    }

    public ParseResult<ISqlExpression> ParseArithmetic_Step1_AdditionOrSubtraction()
    {
        var left = ParseArithmetic_Step2_MultiplicationOrDivision();
        while (PeekSymbolString(1).Equals("+") || PeekSymbolString(1).Equals("-"))
        {
            var op = ReadSymbolString(1);
            var right = ParseArithmetic_Step2_MultiplicationOrDivision();
            left = CreateParseResult(new SqlArithmeticBinaryExpr
            {
                Left = left.ResultValue,
                Operator = op.ToArithmeticOperator(),
                Right = right.ResultValue
            }).To<ISqlExpression>();
        }

        return left;
    }

    public ParseResult<ISqlExpression> ParseArithmetic_Step2_MultiplicationOrDivision()
    {
        var left = ParseArithmetic_Step3_Bitwise();
        while (PeekSymbolString(1).Equals("*") || PeekSymbolString(1).Equals("/"))
        {
            var op= ReadSymbolString(1);
            var right = ParseArithmetic_Step3_Bitwise();
            left = new SqlArithmeticBinaryExpr
            {
                Left = left.ResultValue,
                Operator = op.ToArithmeticOperator(),
                Right = right.ResultValue,
            };
        }
        return left;
    }
    
    public ParseResult<ISqlExpression> ParseArithmetic_Step3_Bitwise()
    {
        var left = ParseArithmetic_Step4_Primary();
        while (PeekSymbolString(1).Equals("&") || PeekSymbolString(1).Equals("|") || PeekSymbolString(1).Equals("^"))
        {
            var op= ReadSymbolString(1);
            var right = ParseArithmetic_Step4_Primary();
            left = new SqlArithmeticBinaryExpr
            {
                Left = left.ResultValue,
                Operator = op.ToArithmeticOperator(),
                Right = right.ResultValue,
            };
        }
        return left; 
    }

    private ParseResult<ISqlExpression> ParseArithmetic_Step4_Primary()
    {
        var startPosition = _text.Position;
        if (Try(ParseValue, out var value))
        {
            if (value.HasError)
            {
                _text.Position = startPosition;
                return value.Error;
            }

            return value.To<ISqlExpression>();
        }

        if (TryMatch("("))
        {
            var subExpr = ParseArithmetic_Step1_AdditionOrSubtraction();
            if (subExpr.HasError)
            {
                return subExpr.Error;
            }
            if (!TryMatch(")"))
            {
                return CreateParseError("InvalidOperationException Mismatched parentheses");
            }
            return new SqlGroup
            {
                Inner = subExpr.ResultValue
            };
        }

        return CreateParseError("InvalidOperationException Unexpected value");
    }
}