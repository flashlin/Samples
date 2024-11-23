using System.Text.RegularExpressions;
using T1.Standard.DesignPatterns;

namespace SqlSharpLit.Common.ParserLit;

public class SqlParser
{
    private const string ConstraintKeyword = "CONSTRAINT";

    private static string[] SqlKeywords =
    [
        "CONSTRAINT", "PRIMARY", "KEY", "UNIQUE"
    ];

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
                yield return rc.Result;
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
            return createTableResult;
        }

        if (Try(ParseSelectStatement, out var selectResult))
        {
            return selectResult;
        }

        if (Try(ParseExecSpAddExtendedProperty, out var execSpAddExtendedPropertyResult))
        {
            return execSpAddExtendedPropertyResult;
        }

        return RaiseParseError("Unknown statement");
    }

    public ParseResult<ISqlExpression> ParseTableForeignKeyExpression()
    {
        if (!TryMatchKeywords("FOREIGN", "KEY"))
        {
            return NoneResult();
        }

        var columnsResult = ParseColumnsAscDesc();
        if (columnsResult.HasError)
        {
            return RaiseParseError(columnsResult.Error);
        }
        var columns = columnsResult.Result.ToList<SqlConstraintColumn>();

        if (!TryMatchKeyword("REFERENCES"))
        {
            return RaiseParseError("Expected REFERENCES");
        }

        var tableName = _text.ReadSqlIdentifier();
        if (tableName.Length == 0)
        {
            return RaiseParseError("Expected reference table name");
        }

        var refColumn = string.Empty;
        if (_text.TryMatch("("))
        {
            refColumn = _text.ReadSqlIdentifier().Word;
            if (!_text.TryMatch(")"))
            {
                return RaiseParseError("Expected )");
            }
        }

        var onDelete = ReferentialAction.NoAction;
        if (TryMatchKeywords("ON", "DELETE"))
        {
            var rc= ParseReferentialAction();
            if (rc.HasError)
            {
                return RaiseParseError(rc.Error);
            }
            onDelete = rc.Result;
        }
        var onUpdate = ReferentialAction.NoAction;
        
        if(TryMatchKeywords("ON", "UPDATE"))
        {
            var rc = ParseReferentialAction();
            if (rc.HasError)
            {
                return RaiseParseError(rc.Error);
            }
            onUpdate = rc.Result;
        }
        
        var notForReplication = TryMatchKeywords("NOT", "FOR", "REPLICATION");
        return CreateParseResult(new SqlTableForeignKeyExpression
        {
            Columns = columns,
            ReferencedTableName = tableName.Word,
            RefColumn = refColumn,
            OnDeleteAction = onDelete,
            OnUpdateAction = onUpdate,
            NotForReplication = notForReplication,
        });
    }

    private ParseResult<ReferentialAction> ParseReferentialAction()
    {
        var result = One(Keywords("NO", "ACTION"), Keywords("CASCADE"), Keywords("SET", "NULL"),
            Keywords("SET", "DEFAULT"))();
        if (result.HasError)
        {
            return RaiseParseError<ReferentialAction>(result.Error);
        }

        var token = (SqlToken)result.Result;
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


    public ParseResult<SqlCollectionExpression> ParseCreateTableColumns()
    {
        var columns = new List<ColumnDefinition>();
        do
        {
            SkipWhiteSpace();
            if (_text.IsPeekIdentifiers(SqlKeywords))
            {
                break;
            }

            var item = _text.ReadSqlIdentifier();
            if (item.Length == 0)
            {
                return EmptyCollectionResult();
            }

            var columnDefinition = ParseColumnTypeDefinition(item);
            if (columnDefinition.HasError)
            {
                return RaiseParseError<SqlCollectionExpression>(columnDefinition.Error);
            }

            if (columnDefinition.Result.SqlType == SqlType.None)
            {
                return RaiseParseError<SqlCollectionExpression>("Expected column definition");
            }

            var column = (ColumnDefinition)columnDefinition.Result;
            ParseColumnConstraints(column);
            columns.Add(column);
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

        return CreateCollectionParseResult(columns);
    }

    public ParseResult<ISqlExpression> ParseCreateTableStatement()
    {
        if (!TryMatchKeywords("CREATE", "TABLE"))
        {
            return NoneResult();
        }

        var tableName = _text.ReadSqlIdentifier();
        if (!_text.TryMatch("("))
        {
            return RaiseParseError("Expected (");
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
                return RaiseParseError(tableColumnsResult.Error);
            }

            var tableColumns = tableColumnsResult.Result.ToList<ColumnDefinition>();
            if (tableColumns.Count > 0)
            {
                createTableStatement.Columns.AddRange(tableColumns);
                continue;
            }

            var tableConstraintsResult = ParseWithComma(ParseTableConstraint);
            if (tableConstraintsResult.HasError)
            {
                return RaiseParseError(tableConstraintsResult.Error);
            }

            var tableConstraints = tableConstraintsResult.Result.Items;
            if (tableConstraints.Count > 0)
            {
                createTableStatement.Constraints.AddRange(tableConstraints);
                continue;
            }

            break;
        }

        if (!_text.TryMatch(")"))
        {
            return RaiseParseError("ParseCreateTableStatement Expected )");
        }

        SkipStatementEnd();

        return CreateParseResult(createTableStatement);
    }

    public ParseResult<ISqlExpression> ParseExecSpAddExtendedProperty()
    {
        if (!TryMatchKeywords("EXEC", "SP_AddExtendedProperty"))
        {
            return NoneResult();
        }

        var parameters = ParseWithComma(ParseParameterValueOrAssignValue);
        if (parameters.HasError)
        {
            return RaiseParseError(parameters.Error);
        }

        if (parameters.Result.Items.Count != 8)
        {
            return RaiseParseError("Expected 8 parameters");
        }

        var p = parameters.Result.ToList<SqlParameterValue>();

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

    public ParseResult<ISqlExpression> ParseSelectStatement()
    {
        if (!TryMatchKeyword("SELECT"))
        {
            return NoneResult();
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
                return RaiseParseError("Expected column name");
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
                return RaiseParseError(leftExpr.Error);
            }

            if (leftExpr.Result.SqlType == SqlType.None)
            {
                return RaiseParseError("Expected left expression");
            }

            var operation = _text.ReadSymbols().Word;
            var rightExpr = ParseValue();
            if (rightExpr.HasError)
            {
                return RaiseParseError(rightExpr.Error);
            }

            if (rightExpr.Result.SqlType == SqlType.None)
            {
                return RaiseParseError("Expected right expression");
            }

            selectStatement.Where = new SqlWhereExpression()
            {
                Left = leftExpr.Result,
                Operation = operation,
                Right = rightExpr.Result
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

    public bool Try(Func<ParseResult<ISqlExpression>> parseFunc, out ParseResult<ISqlExpression> result)
    {
        var localResult = parseFunc();
        if (localResult.HasError)
        {
            result = localResult;
            return true;
        }

        if (localResult.Result == SqlNoneExpression.Default)
        {
            result = localResult;
            return false;
        }

        result = localResult;
        return true;
    }

    private ParseResult<SqlCollectionExpression> CreateCollectionParseResult<T>(IEnumerable<T> result)
    {
        return new ParseResult<SqlCollectionExpression>(new SqlCollectionExpression()
        {
            Items = result.Cast<ISqlExpression>().ToList()
        });
    }

    private ParseResult<ISqlExpression> CreateParseResult<T>(T result)
        where T : ISqlExpression
    {
        return new ParseResult<ISqlExpression>(result);
    }

    private ParseResult<SqlCollectionExpression> EmptyCollectionResult()
    {
        return new ParseResult<SqlCollectionExpression>(new SqlCollectionExpression());
    }


    private Func<ParseResult<ISqlExpression>> Keywords(params string[] keywords)
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
        return new ParseResult<ISqlExpression>(SqlNoneExpression.Default);
    }

    private ISqlExpression Optional(Func<ParseResult<ISqlExpression>> parseFn)
    {
        var result = parseFn();
        if (result.HasResult && result.Result.SqlType != SqlType.None)
        {
            return result.Result;
        }

        return SqlNoneExpression.Default;
    }

    private Func<ParseResult<ISqlExpression>> Or(params Func<ParseResult<ISqlExpression>>[] parseFnList)
    {
        return () =>
        {
            foreach (var parseFn in parseFnList)
            {
                var rc = parseFn();
                if (rc.HasResult && rc.Result.SqlType != SqlType.None)
                {
                    return rc;
                }

                if (rc.HasError)
                {
                    return rc;
                }
            }

            return NoneResult();
        };
    }

    private Func<ParseResult<ISqlExpression>> One(params Func<ParseResult<ISqlExpression>>[] parseFnList)
    {
        return () =>
        {
            var rc = Or(parseFnList)();
            if (rc.Result.SqlType != SqlType.None)
            {
                return rc;
            }

            if (rc.HasError)
            {
                return rc;
            }

            return RaiseParseError("Expected one of the options");
        };
    }

    private ParseResult<ColumnDefinition> ParseColumnConstraints(ColumnDefinition column)
    {
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
                    return RaiseParseError<ColumnDefinition>(identityResult.Error);
                }

                column.Identity = (SqlIdentity)identityResult.Result;
                continue;
            }

            if (Try(ParseDefaultValue, out var defaultValue))
            {
                if (identityResult.HasError)
                {
                    return RaiseParseError<ColumnDefinition>(identityResult.Error);
                }

                column.Constraints.Add((SqlConstraintPrimaryKeyOrUnique)defaultValue.Result);
                continue;
            }

            var constraintStartPosition = _text.Position;
            if (_text.TryMatch(ConstraintKeyword))
            {
                var constraintName = _text.ReadSqlIdentifier();
                if (Try(ParseDefaultValue, out var constraintDefaultValue))
                {
                    if (identityResult.HasError)
                    {
                        return RaiseParseError<ColumnDefinition>(identityResult.Error);
                    }

                    var subConstraint = (SqlConstraintPrimaryKeyOrUnique)constraintDefaultValue.Result;
                    subConstraint.ConstraintName = constraintName.Word;
                    column.Constraints.Add(subConstraint);
                    continue;
                }

                _text.Position = constraintStartPosition;
                var columnConstraint = ParseTableConstraint();
                if (columnConstraint.HasError)
                {
                    return RaiseParseError<ColumnDefinition>(columnConstraint.Error);
                }

                if (columnConstraint.Result.SqlType != SqlType.None)
                {
                    var t = (SqlConstraintPrimaryKeyOrUnique)columnConstraint.Result;
                    column.Constraints.Add(t);
                }

                return RaiseParseError<ColumnDefinition>("Expect Constraint DEFAULT");
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

        return ToParseResult(column);
    }

    private ParseResult<ISqlExpression> ParseColumnTypeDefinition(TextSpan columnNameSpan)
    {
        var column = new ColumnDefinition
        {
            ColumnName = columnNameSpan.Word,
            DataType = _text.ReadSqlIdentifier().Word
        };

        var dataLength1 = string.Empty;
        var dataLength2 = string.Empty;
        if (_text.TryMatch("("))
        {
            if (_text.TryMatchIgnoreCaseKeyword("MAX"))
            {
                column.Size = "MAX";
                _text.Match(")");
                return CreateParseResult(column);
            }

            dataLength1 = _text.ReadNumber().Word;
            dataLength2 = string.Empty;
            if (_text.PeekChar() == ',')
            {
                _text.ReadChar();
                dataLength2 = _text.ReadNumber().Word;
            }

            if (!_text.TryMatch(")"))
            {
                return RaiseParseError("Expected )");
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

    private ParseResult<ISqlExpression> ParseDefaultValue()
    {
        if (!TryMatchKeyword("DEFAULT"))
        {
            return NoneResult();
        }

        TextSpan defaultValue;
        if (_text.TryMatch("("))
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

        defaultValue = _text.ReadNumber();
        return CreateParseResult(new SqlConstraintPrimaryKeyOrUnique
        {
            ConstraintName = string.Empty,
            DefaultValue = defaultValue.Word,
        });
    }

    private ParseResult<ISqlExpression> ParseIdentity()
    {
        if (!_text.TryMatch("IDENTITY"))
        {
            return NoneResult();
        }

        var sqlIdentity = new SqlIdentity
        {
            Seed = 1,
            Increment = 1
        };
        if (_text.TryMatch("("))
        {
            sqlIdentity.Seed = long.Parse(_text.ReadNumber().Word);
            _text.Match(",");
            sqlIdentity.Increment = int.Parse(_text.ReadNumber().Word);
            _text.Match(")");
        }

        return CreateParseResult(sqlIdentity);
    }

    private ParseResult<ISqlExpression> ParseIntValue()
    {
        if (_text.Try(_text.ReadNumber, out var number))
        {
            return CreateParseResult(new SqlIntValueExpression
            {
                Value = number.Word
            });
        }

        return NoneResult();
    }

    private ParseResult<ISqlExpression> ParseKeywords(params string[] keywords)
    {
        if (TryMatchKeywords(keywords))
        {
            return CreateParseResult(new SqlToken
            {
                Value = string.Join(" ", keywords)
            });
        }

        return NoneResult();
    }

    private ParseResult<ISqlExpression> ParseParameterAssignValue()
    {
        SkipWhiteSpace();
        if (!_text.Try(_text.ReadSqlIdentifier, out var name))
        {
            return NoneResult();
        }

        if (!_text.TryMatch("="))
        {
            return RaiseParseError("Expected =");
        }

        if (!_text.Try(_text.ReadSqlQuotedString, out var nameValue))
        {
            return RaiseParseError($"Expected @name value, but got {_text.PreviousWord().Word}");
        }

        return CreateParseResult(new SqlParameterValue
        {
            Name = name.Word,
            Value = nameValue.Word
        });
    }

    private ParseResult<ISqlExpression> ParseParameterValue()
    {
        SkipWhiteSpace();
        var startPosition = _text.Position;
        var valueResult = ParseValue();
        if (valueResult.HasError)
        {
            _text.Position = startPosition;
            return RaiseParseError(valueResult.Error);
        }

        if (valueResult.Result.SqlType == SqlType.None)
        {
            return NoneResult();
        }

        if (_text.Peek(_text.ReadSymbols).Word == "=")
        {
            _text.Position = startPosition;
            return NoneResult();
        }

        return CreateParseResult(new SqlParameterValue
        {
            Name = string.Empty,
            Value = ((ISqlValue)valueResult.Result).Value
        });
    }

    private ParseResult<ISqlExpression> ParseParameterValueOrAssignValue()
    {
        var rc1 = ParseParameterValue();
        if (rc1.HasError)
        {
            return RaiseParseError(rc1.Error);
        }

        if (rc1.HasResult && rc1.Result.SqlType != SqlType.None)
        {
            return rc1;
        }

        var rc2 = ParseParameterAssignValue();
        if (rc2.HasError)
        {
            return RaiseParseError(rc2.Error);
        }

        if (rc2.HasResult && rc2.Result.SqlType != SqlType.None)
        {
            return rc2;
        }

        return NoneResult();
    }

    private ParseResult<SqlCollectionExpression> ParseParenthesesWithComma<T>(Func<ParseResult<T>> parseElemFn)
        where T : ISqlExpression
    {
        if (!_text.TryMatch("("))
        {
            return RaiseParseError<SqlCollectionExpression>("Expected (");
        }

        var elements = ParseWithComma(parseElemFn);
        if (elements.HasError)
        {
            return RaiseParseError<SqlCollectionExpression>(elements.Error);
        }

        if (!_text.TryMatch(")"))
        {
            return RaiseParseError<SqlCollectionExpression>("Expected )");
        }

        return elements;
    }

    private ParseResult<ISqlExpression> ParsePrimaryKeyOrUnique()
    {
        var sqlConstraint = new SqlConstraintPrimaryKeyOrUnique();
        var primaryKeyOrUniqueToken = Optional(Or(Keywords("PRIMARY", "KEY"), Keywords("UNIQUE")));
        if (primaryKeyOrUniqueToken.SqlType != SqlType.None)
        {
            sqlConstraint.ConstraintType = ((SqlToken)primaryKeyOrUniqueToken).Value;
        }

        if (string.IsNullOrEmpty(sqlConstraint.ConstraintType))
        {
            return NoneResult();
        }

        var clusteredToken = Optional(Or(Keywords("CLUSTERED"), Keywords("NONCLUSTERED")));
        if (clusteredToken != SqlNoneExpression.Default)
        {
            sqlConstraint.Clustered = ((SqlToken)clusteredToken).Value;
        }

        var columnsResult = ParseColumnsAscDesc();
        if (columnsResult.HasError)
        {
            return RaiseParseError(columnsResult.Error);
        }
        sqlConstraint.Columns = columnsResult.Result.ToList<SqlConstraintColumn>();
        return CreateParseResult(sqlConstraint);
    }

    private ParseResult<SqlCollectionExpression> ParseColumnsAscDesc()
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

    private ParseResult<ISqlExpression> ParseTableConstraint()
    {
        var constraintName = string.Empty;
        if (TryMatchKeyword(ConstraintKeyword))
        {
            constraintName = _text.ReadSqlIdentifier().Word;
        }
        
        var tablePrimaryKeyOrUniqueExpr = ParsePrimaryKeyOrUniqueExpr();
        if (tablePrimaryKeyOrUniqueExpr.HasError)
        {
            return RaiseParseError(tablePrimaryKeyOrUniqueExpr.Error);
        }
        if (tablePrimaryKeyOrUniqueExpr.Result.SqlType != SqlType.None)
        {
            ((SqlConstraintPrimaryKeyOrUnique)tablePrimaryKeyOrUniqueExpr.Result).ConstraintName = constraintName;
            return tablePrimaryKeyOrUniqueExpr;
        }
        
        var tableForeignKeyExpr = ParseTableForeignKeyExpression();
        if (tableForeignKeyExpr.HasError)
        {
            RaiseParseError(tableForeignKeyExpr.Error);
        }
        if(tableForeignKeyExpr.Result.SqlType != SqlType.None)
        {
            ((SqlTableForeignKeyExpression)tableForeignKeyExpr.Result).ConstraintName = constraintName;
            return tableForeignKeyExpr;
        }
        
        return NoneResult();
    }

    private ParseResult<ISqlExpression> ParsePrimaryKeyOrUniqueExpr()
    {
        var primaryKeyOrUnique = ParsePrimaryKeyOrUnique();
        if (primaryKeyOrUnique.HasError)
        {
            return RaiseParseError(primaryKeyOrUnique.Error);
        }
        
        var sqlConstraint = new SqlConstraintPrimaryKeyOrUnique();
        var hasSetting = false;
        if (primaryKeyOrUnique.Result.SqlType != SqlType.None)
        {
            var subConstraint = (SqlConstraintPrimaryKeyOrUnique)primaryKeyOrUnique.Result;
            sqlConstraint.ConstraintType = subConstraint.ConstraintType;
            sqlConstraint.Clustered = subConstraint.Clustered;
            sqlConstraint.Columns = subConstraint.Columns;
            hasSetting = true;
        }

        if (TryMatchKeyword("WITH"))
        {
            var togglesResult = ParseParenthesesWithComma(ParseWithToggle);
            if (togglesResult.HasError)
            {
                return RaiseParseError(togglesResult.Error);
            }

            sqlConstraint.WithToggles = togglesResult.Result.ToList<SqlToggle>();
            hasSetting = true;
        }

        if (TryMatchKeyword("ON"))
        {
            sqlConstraint.On = _text.ReadSqlIdentifier().Word;
            hasSetting = true;
        }

        if (Try(ParseIdentity, out var identityResult))
        {
            if (identityResult.HasError)
            {
                return RaiseParseError(identityResult.Error);
            }

            sqlConstraint.Identity = (SqlIdentity)identityResult.Result;
            hasSetting = true;
        }

        if (!hasSetting)
        {
            return NoneResult();
        }
        return CreateParseResult(sqlConstraint);
    }

    private ParseResult<ISqlExpression> ParseTableName()
    {
        if (_text.Try(_text.ReadIdentifier, out var fieldName))
        {
            return CreateParseResult(new SqlFieldExpression()
            {
                FieldName = fieldName.Word
            });
        }

        return NoneResult();
    }

    private ParseResult<ISqlExpression> ParseValue()
    {
        if (Try(ParseIntValue, out var number))
        {
            return number;
        }

        if (_text.Try(_text.ReadSqlQuotedString, out var quotedString))
        {
            return CreateParseResult(new SqlStringValue
            {
                Value = quotedString.Word
            });
        }

        if (Try(ParseTableName, out var tableName))
        {
            return tableName;
        }

        return NoneResult();
    }

    private ParseResult<SqlCollectionExpression> ParseWithComma<T>(Func<ParseResult<T>> parseElemFn)
        where T : ISqlExpression
    {
        var elements = new List<T>();
        do
        {
            var elem = parseElemFn();
            if (elem is { HasResult: true, Result.SqlType: SqlType.None })
            {
                break;
            }

            if (elem.HasError)
            {
                return RaiseParseError<SqlCollectionExpression>(elem.Error);
            }

            elements.Add(elem.Result);
            if (_text.PeekChar() != ',')
            {
                break;
            }

            _text.ReadChar();
        } while (!_text.IsEnd());

        return CreateCollectionParseResult(elements);
    }

    private ParseResult<ISqlExpression> ParseWithToggle()
    {
        var startPosition = _text.Position;
        var toggleName = _text.ReadSqlIdentifier();
        if (toggleName.Length == 0)
        {
            _text.Position = startPosition;
            return NoneResult();
        }

        var toggle = new SqlToggle
        {
            ToggleName = toggleName.Word
        };

        if (!_text.TryMatch("="))
        {
            _text.Position = startPosition;
            return RaiseParseError("Expected toggleName =");
        }

        if (_text.Try(_text.ReadNumber, out var number))
        {
            toggle.Value = number.Word;
            return CreateParseResult(toggle);
        }

        toggle.Value = _text.ReadSqlIdentifier().Word;
        return CreateParseResult(toggle);
    }

    private bool PeekMatchSymbol(string symbol)
    {
        SkipWhiteSpace();
        return _text.PeekMatchSymbol(symbol);
    }

    private ParseResult<ISqlExpression> RaiseParseError(string error)
    {
        return new ParseResult<ISqlExpression>(new ParseError(error)
        {
            Offset = _text.Position
        });
    }

    private ParseResult<ISqlExpression> RaiseParseError(string error, int startOffset)
    {
        var errorPosition = _text.Position;
        _text.Position = startOffset;
        return new ParseResult<ISqlExpression>(new ParseError(error)
        {
            Offset = errorPosition
        });
    }

    private ParseResult<ISqlExpression> RaiseParseError(ParseError innerError)
    {
        return new ParseResult<ISqlExpression>(innerError);
    }

    private ParseResult<T> RaiseParseError<T>(ParseError innerError)
    {
        return new ParseResult<T>(innerError);
    }

    private ParseResult<T> RaiseParseError<T>(string error)
    {
        return new ParseResult<T>(new ParseError(error)
        {
            Offset = _text.Position
        });
    }

    private void ReadNonWhiteSpace()
    {
        var sqlIdentifier = _text.ReadSqlIdentifier();
        if (sqlIdentifier.Length > 0)
        {
            return;
        }

        var sqlString = _text.ReadSqlQuotedString();
        if (sqlString.Length > 0)
        {
            return;
        }

        var sqlNumber = _text.ReadNumber();
        if (sqlNumber.Length > 0)
        {
            return;
        }

        var sqlSymbol = _text.ReadSymbols();
        if (sqlSymbol.Length > 0)
        {
            return;
        }

        _text.NextChar();
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

    private ParseResult<T> ToParseResult<T>(T result)
    {
        return new ParseResult<T>(result);
    }

    private bool TryMatchKeywords(params string[] keywords)
    {
        SkipWhiteSpace();
        return _text.TryMatchesIgnoreCase(keywords);
    }

    private bool TryMatchKeyword(string expected)
    {
        SkipWhiteSpace();
        return _text.TryMatchIgnoreCaseKeyword(expected);
    }

    private bool TryMatchPrimaryKeyOrUnique(SqlConstraintPrimaryKeyOrUnique sqlConstraintPrimaryKeyOrUnique)
    {
        if (TryMatchKeyword("UNIQUE"))
        {
            sqlConstraintPrimaryKeyOrUnique.ConstraintType = "UNIQUE";
            return true;
        }

        if (TryMatchKeywords("PRIMARY", "KEY"))
        {
            sqlConstraintPrimaryKeyOrUnique.ConstraintType = "PRIMARY KEY";
            return true;
        }

        return false;
    }

    private bool TryParseSqlIdentity(ColumnDefinition column, out Either<ColumnDefinition, ParseError> result)
    {
        if (!_text.TryMatch("IDENTITY"))
        {
            result = new Either<ColumnDefinition, ParseError>(column);
            return false;
        }

        var sqlIdentity = new SqlIdentity
        {
            Seed = 1,
            Increment = 1
        };
        if (_text.TryMatch("("))
        {
            sqlIdentity.Seed = long.Parse(_text.ReadNumber().Word);
            _text.Match(",");
            sqlIdentity.Increment = int.Parse(_text.ReadNumber().Word);
            _text.Match(")");
        }

        column.Identity = sqlIdentity;
        result = new Either<ColumnDefinition, ParseError>(column);
        return true;
    }

    private bool TryPeekKeyword(string expected)
    {
        SkipWhiteSpace();
        var tmpPosition = _text.Position;
        var isSuccess = _text.TryMatchIgnoreCaseKeyword(expected);
        _text.Position = tmpPosition;
        return isSuccess;
    }
}

public class ParseResult<T>
{
    public ParseResult(T result)
    {
        HasResult = true;
        Result = result;
    }

    public ParseResult(ParseError error)
    {
        HasError = true;
        Error = error;
    }

    public T Result { get; set; }
    public bool HasResult { get; set; }
    public ParseError Error { get; set; } = ParseError.Empty;
    public bool HasError { get; set; }
}