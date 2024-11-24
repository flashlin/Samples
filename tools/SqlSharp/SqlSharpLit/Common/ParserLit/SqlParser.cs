using System.Text.RegularExpressions;

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
            if (rc.HasResult)
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
            if(columnDefinition.Result!=null)
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

    public ParseResult<CreateTableStatement> ParseCreateTableStatement()
    {
        if (!TryMatchKeywords("CREATE", "TABLE"))
        {
            return NoneResult<CreateTableStatement>();
        }

        var tableName = _text.ReadSqlIdentifier();
        if (!TryMatch("("))
        {
            return CreateParseError("Expected (");
        }

        var createTableStatement = new CreateTableStatement()
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

    public ParseResult<SqlSpAddExtendedProperty> ParseExecSpAddExtendedProperty()
    {
        if (!TryMatchKeywords("EXEC", "SP_AddExtendedProperty"))
        {
            return NoneResult<SqlSpAddExtendedProperty>();
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

        var sqlSpAddExtendedProperty = new SqlSpAddExtendedProperty
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
        if (!TryMatchKeywords("FOREIGN", "KEY"))
        {
            return NoneResult<SqlConstraintForeignKey>();
        }

        var columnsResult = ParseColumnsAscDesc();
        if (columnsResult.HasError)
        {
            return columnsResult.Error;
        }
        var columns = columnsResult.ResultValue;
        if (!TryMatchKeyword("REFERENCES"))
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
        if (TryMatchKeywords("ON", "DELETE"))
        {
            var rc = ParseReferentialAction();
            if (rc.HasError)
            {
                return rc.Error;
            }

            onDelete = rc.Result;
        }

        var onUpdate = ReferentialAction.NoAction;

        if (TryMatchKeywords("ON", "UPDATE"))
        {
            var rc = ParseReferentialAction();
            if (rc.HasError)
            {
                return rc.Error;
            }

            onUpdate = rc.Result;
        }

        var notForReplication = TryMatchKeywords("NOT", "FOR", "REPLICATION");
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

    public ParseResult<SelectStatement> ParseSelectStatement()
    {
        if (!TryMatchKeyword("SELECT"))
        {
            return NoneResult<SelectStatement>();
        }

        var columns = new List<ISelectColumnExpression>();
        do
        {
            if (_text.Try(_text.ReadIdentifier, out var fieldName))
            {
                columns.Add(new SelectColumn()
                {
                    ColumnName = fieldName.Word
                });
            }
            else
            {
                return CreateParseError("Expected column name");
            }

            if (_text.PeekChar() != ',')
            {
                break;
            }

            _text.ReadChar();
        } while (!_text.IsEnd());

        var selectStatement = new SelectStatement
        {
            Columns = columns
        };

        if (TryMatchKeyword("FROM"))
        {
            var tableName = _text.ReadIdentifier().Word;
            selectStatement.From = new SelectFrom()
            {
                FromTableName = tableName
            };
        }

        if (TryMatchKeyword("WHERE"))
        {
            var leftExpr = ParseValue();
            if (leftExpr.HasError)
            {
                return leftExpr.Error;
            }
            if (leftExpr.Result==null)
            {
                return CreateParseError("Expected left expression");
            }

            var operation = _text.ReadSymbols().Word;
            var rightExpr = ParseValue();
            if (rightExpr.HasError)
            {
                return rightExpr.Error;
            }
            if (rightExpr.Result==null)
            {
                return CreateParseError("Expected right expression");
            }

            selectStatement.Where = new SqlWhereExpression()
            {
                Left = leftExpr.ResultValue,
                Operation = operation,
                Right = rightExpr.ResultValue
            };
        }

        SkipStatementEnd();
        return CreateParseResult(selectStatement);
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

    private bool IsAny<T>(params Func<ParseResult<T>>[] parseFnList)
    {
        var span = Or(parseFnList)();
        if (span.HasError)
        {
            return false;
        }

        return span.Result != null;
    }

    private Func<ParseResult<SqlToken>> Keywords(params string[] keywords)
    {
        return () => ParseKeywords(keywords);
    }

    private void MatchString(string expected)
    {
        SkipWhiteSpace();
        _text.Match(expected);
    }

    private ParseResult<ISqlExpression> NoneResult()
    {
        return new ParseResult<ISqlExpression>(default(ISqlExpression));
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
            if (rc.Result!=null)
            {
                return rc;
            }
            if (rc.HasError)
            {
                return rc;
            }
            return CreateParseError("Expected one of the options");
        };
    }

    private T? Optional<T>(Func<ParseResult<T>> parseFn)
    {
        var result = parseFn();
        return result.Result;
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
                    return CreateParseResult(rc.ResultValue);
                }
            }
            return NoneResult<T>();
        };
    }

    private ParseResult<ColumnDefinition> ParseColumnConstraints(ColumnDefinition column)
    {
        var comma = PeekKeywords(",")();
        if(comma.Result!=null)
        {
            return NoneResult<ColumnDefinition>();
        }
            
        do
        {
            var startPosition = _text.Position;
            if (TryMatchKeywords("PRIMARY", "KEY"))
            {
                // 最後一個column 有可能沒有逗號 又寫 Table Constraint 的話會被誤判, 所以要檢查是否有 CLUSTERED 
                if (TryMatchKeyword("CLUSTERED"))
                {
                    _text.Position = startPosition;
                    break;
                }

                column.IsPrimaryKey = true;
                continue;
            }

            if (Try(ParseIdentity, out var identityResult))
            {
                if (identityResult.HasError)
                {
                    return identityResult.Error;
                }
                column.Identity = identityResult.ResultValue;
                continue;
            }

            if (Try(ParseDefaultValue, out var defaultValue))
            {
                if (identityResult.HasError)
                {
                    return identityResult.Error;
                }
                column.Constraints.Add(defaultValue.ResultValue);
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
                    column.Constraints.Add(subConstraint);
                    continue;
                }

                _text.Position = constraintStartPosition;
                var columnConstraint = ParseTableConstraint();
                if (columnConstraint.HasError)
                {
                    return columnConstraint.Error;
                }

                if (columnConstraint.Result==null)
                {
                    return CreateParseError("Expect Constraint DEFAULT");
                }
                column.Constraints.Add(columnConstraint.Result);
            }

            if (TryMatchKeywords("NOT", "FOR", "REPLICATION"))
            {
                column.NotForReplication = true;
                continue;
            }

            if (TryMatchKeywords("NOT", "NULL"))
            {
                column.IsNullable = false;
                continue;
            }

            if (TryMatchKeywords("NULL"))
            {
                column.IsNullable = true;
                continue;
            }

            break;
        } while (true);

        return CreateParseResult(column);
    }

    private ParseResult<ColumnDefinition> ParseColumnDefinition()
    {
        var startPosition = _text.Position;
        if (!TryReadSqlIdentifier(out var columnNameSpan))
        {
            return NoneResult<ColumnDefinition>();
        }
        
        var columnDefinition = ParseColumnTypeDefinition(columnNameSpan);
        if (columnDefinition.HasError)
        {
            return columnDefinition.Error;
        }
        if (columnDefinition.Result==null)
        {
            _text.Position = startPosition;
            return NoneResult<ColumnDefinition>();
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
            if (TryMatchKeyword("ASC"))
            {
                order = "ASC";
            }
            else if (TryMatchKeyword("DESC"))
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

    private ParseResult<ColumnDefinition> ParseColumnTypeDefinition(TextSpan columnNameSpan)
    {
        var column = new ColumnDefinition
        {
            ColumnName = columnNameSpan.Word,
            DataType = ReadSqlIdentifier().Word
        };

        var dataLength1 = string.Empty;
        var dataLength2 = string.Empty;
        if (TryMatch("("))
        {
            if (_text.TryMatchIgnoreCaseKeyword("MAX"))
            {
                column.Size = "MAX";
                _text.Match(")");
                return CreateParseResult(column);
            }

            dataLength1 = _text.ReadInt().Word;
            dataLength2 = string.Empty;
            if (_text.PeekChar() == ',')
            {
                _text.ReadChar();
                dataLength2 = _text.ReadInt().Word;
            }

            if (!TryMatch(")"))
            {
                return CreateParseError("Expected )");
            }
        }

        if (!string.IsNullOrEmpty(dataLength1))
        {
            column.Size = dataLength1;
        }

        if (!string.IsNullOrEmpty(dataLength2))
        {
            column.Scale = int.Parse(dataLength2);
        }

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

        if (!TryMatchKeyword("AS"))
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

        var persist = TryMatchKeyword("PERSISTED");
        var notNull = TryMatchKeywords("NOT", "NULL");

        return CreateParseResult(new SqlComputedColumnDefinition
        {
            ColumnName = columnNameSpan.Word,
            Expression = computedColumnExpressionSpan.Word,
            IsPersisted = persist,
            IsNotNull = notNull
        });
    }

    private ParseResult<SqlConstraintPrimaryKeyOrUnique> ParseDefaultValue()
    {
        if (!TryMatchKeyword("DEFAULT"))
        {
            return NoneResult<SqlConstraintPrimaryKeyOrUnique>();
        }

        TextSpan defaultValue;
        if (TryMatch("("))
        {
            defaultValue = _text.ReadUntilRightParenthesis();
            _text.Match(")");
            return CreateParseResult(new SqlConstraintPrimaryKeyOrUnique
            {
                ConstraintName = string.Empty,
                DefaultValue = defaultValue.Word
            });
        }

        var nullValue = _text.PeekIdentifier("NULL");
        if (nullValue.Length > 0)
        {
            _text.ReadIdentifier();
            return CreateParseResult(new SqlConstraintPrimaryKeyOrUnique
            {
                ConstraintName = string.Empty,
                DefaultValue = nullValue.Word
            });
        }

        if (_text.Try(_text.ReadSqlIdentifier, out var funcName))
        {
            _text.Match("(");
            var funcArgs = _text.ReadUntilRightParenthesis();
            _text.Match(")");
            defaultValue = new TextSpan
            {
                Word = $"{funcName.Word}({funcArgs.Word})",
                Offset = funcName.Offset,
                Length = funcName.Length + funcArgs.Length + 2
            };
            return CreateParseResult(new SqlConstraintPrimaryKeyOrUnique
            {
                ConstraintName = string.Empty,
                DefaultValue = defaultValue.Word,
            });
        }

        if (_text.Try(_text.ReadSqlQuotedString, out var quotedString))
        {
            return CreateParseResult(new SqlConstraintPrimaryKeyOrUnique
            {
                ConstraintName = string.Empty,
                DefaultValue = quotedString.Word,
            });
        }

        if (_text.Try(_text.ReadSqlDate, out var date))
        {
            return CreateParseResult(new SqlConstraintPrimaryKeyOrUnique
            {
                ConstraintName = string.Empty,
                DefaultValue = date.Word,
            });
        }

        if (_text.Try(_text.ReadNegativeNumber, out var negativeNumber))
        {
            return CreateParseResult(new SqlConstraintPrimaryKeyOrUnique
            {
                ConstraintName = string.Empty,
                DefaultValue = negativeNumber.Word,
            });
        }

        if (_text.Try(_text.ReadFloat, out var floatNumber))
        {
            return CreateParseResult(new SqlConstraintPrimaryKeyOrUnique
            {
                ConstraintName = string.Empty,
                DefaultValue = floatNumber.Word,
            });
        }

        defaultValue = _text.ReadInt();
        return CreateParseResult(new SqlConstraintPrimaryKeyOrUnique
        {
            ConstraintName = string.Empty,
            DefaultValue = defaultValue.Word,
        });
    }

    private ParseResult<SqlIdentity> ParseIdentity()
    {
        if (!_text.TryMatch("IDENTITY"))
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
            _text.Match(",");
            sqlIdentity.Increment = int.Parse(_text.ReadInt().Word);
            _text.Match(")");
        }

        return CreateParseResult(sqlIdentity);
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

    private ParseResult<SqlToken> ParseKeywords(params string[] keywords)
    {
        if (TryMatchKeywords(keywords))
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

        if (valueResult.Result==null)
        {
            return NoneResult<SqlParameterValue>();
        }

        if (_text.Peek(_text.ReadSymbols).Word == "=")
        {
            _text.Position = startPosition;
            return NoneResult<SqlParameterValue>();
        }

        return CreateParseResult(new SqlParameterValue
        {
            Name = string.Empty,
            Value = valueResult.ResultValue.Value
        });
    }

    private ParseResult<SqlParameterValue> ParseParameterValueOrAssignValue()
    {
        var rc1 = ParseParameterValue();
        if (rc1.HasError)
        {
            return rc1.Error;
        }

        if (rc1.HasResult && rc1.Result!=null)
        {
            return rc1;
        }

        var rc2 = ParseParameterAssignValue();
        if (rc2.HasError)
        {
            return rc2.Error;
        }

        if (rc2.HasResult && rc2.Result!=null)
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
        var primaryKeyOrUniqueToken = Optional(Or(Keywords("PRIMARY", "KEY"), Keywords("UNIQUE")));
        if (primaryKeyOrUniqueToken!=null)
        {
            sqlConstraint.ConstraintType = primaryKeyOrUniqueToken.Value;
        }

        if (string.IsNullOrEmpty(sqlConstraint.ConstraintType))
        {
            return NoneResult<SqlConstraintPrimaryKeyOrUnique>();
        }

        var clusteredToken = Optional(Or(Keywords("CLUSTERED"), Keywords("NONCLUSTERED")));
        if (clusteredToken != null)
        {
            sqlConstraint.Clustered = clusteredToken.Value;
        }

        var columnsResult = ParseColumnsAscDesc();
        if (columnsResult.HasError)
        {
            return columnsResult.Error;
        }

        sqlConstraint.Columns = columnsResult.ResultValue;
        return CreateParseResult(sqlConstraint);
    }

    private ParseResult<SqlConstraintPrimaryKeyOrUnique> ParsePrimaryKeyOrUniqueExpression()
    {
        var primaryKeyOrUniqueResult = ParsePrimaryKeyOrUnique();
        if (primaryKeyOrUniqueResult.HasError)
        {
            return primaryKeyOrUniqueResult;
        }

        if (primaryKeyOrUniqueResult.Result==null)
        {
            return NoneResult<SqlConstraintPrimaryKeyOrUnique>();
        }

        var sqlConstraint = primaryKeyOrUniqueResult.ResultValue;

        if (TryMatchKeyword("WITH"))
        {
            var togglesResult = ParseParenthesesWithComma(ParseWithToggle);
            if (togglesResult.HasError)
            {
                return togglesResult.Error;
            }
            sqlConstraint.WithToggles = togglesResult.ResultValue;
        }

        if (TryMatchKeyword("ON"))
        {
            sqlConstraint.On = _text.ReadSqlIdentifier().Word;
        }

        if (Try(ParseIdentity, out var identityResult))
        {
            if (identityResult.HasError)
            {
                return identityResult.Error;
            }

            sqlConstraint.Identity = identityResult.ResultValue;
        }

        return CreateParseResult(sqlConstraint);
    }

    private ParseResult<ReferentialAction> ParseReferentialAction()
    {
        var result = One(Keywords("NO", "ACTION"), Keywords("CASCADE"), Keywords("SET", "NULL"),
            Keywords("SET", "DEFAULT"))();
        if (result.HasError)
        {
            return result.Error;
        }

        var token = result.ResultValue;
        var action = token.Value.ToUpper() switch
        {
            "NO ACTION" => ReferentialAction.NoAction,
            "CASCADE" => ReferentialAction.Cascade,
            "SET NULL" => ReferentialAction.SetNull,
            "SET DEFAULT" => ReferentialAction.SetDefault,
            _ => ReferentialAction.NoAction
        };
        return new ParseResult<ReferentialAction>(action);
    }

    private ParseResult<ISqlExpression> ParseTableConstraint()
    {
        var constraintName = string.Empty;
        if (TryMatchKeyword(ConstraintKeyword))
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

        if (tableForeignKeyExpr.Result!=null)
        {
            tableForeignKeyExpr.Result.ConstraintName = constraintName;
            return tableForeignKeyExpr.Result;
        }

        return NoneResult<ISqlExpression>();
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

    private ParseResult<ISqlValue> ParseValue()
    {
        if (_text.Try(_text.ReadFloat, out var floatNumber))
        {
            return CreateParseResult<ISqlValue>(new SqlValue
            {
                Value = floatNumber.Word
            });
        }

        if (Try(ParseIntValue, out var number))
        {
            return CreateParseResult<ISqlValue>(number.ResultValue);
        }

        if (_text.Try(_text.ReadSqlQuotedString, out var quotedString))
        {
            return CreateParseResult<ISqlValue>(new SqlValue
            {
                Value = quotedString.Word
            });
        }

        if (Try(ParseTableName, out var tableName))
        {
            return CreateParseResult<ISqlValue>(tableName.ResultValue);
        }

        return NoneResult<ISqlValue>();
    }

    private ParseResult<List<T>> ParseWithComma<T>(Func<ParseResult<T>> parseElemFn)
    {
        var elements = new List<T>();
        do
        {
            var elem = parseElemFn();
            if (elem is { HasResult: true, Result: null})
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

    private bool TryMatchKeyword(string expected)
    {
        SkipWhiteSpace();
        return _text.TryMatchIgnoreCaseKeyword(expected);
    }

    private bool TryMatchKeywords(params string[] keywords)
    {
        SkipWhiteSpace();
        return _text.TryMatchesIgnoreCase(keywords);
    }

    private bool TryPeekKeyword(string expected)
    {
        SkipWhiteSpace();
        var tmpPosition = _text.Position;
        var isSuccess = _text.TryMatchIgnoreCaseKeyword(expected);
        _text.Position = tmpPosition;
        return isSuccess;
    }

    private bool TryReadSqlIdentifier(out TextSpan result)
    {
        SkipWhiteSpace();
        return _text.Try(_text.ReadSqlIdentifier, out result);
    }
}