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
            if (rc.IsLeft)
            {
                yield return rc.LeftValue;
            }
            else
            {
                _text.ReadUntil(c => c == '\n');
            }
        }
    }

    public Either<ISqlExpression, ParseError> Parse()
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


    public Either<SqlCollectionExpression, ParseError> ParseCreateTableColumns()
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
            if (columnDefinition.IsRight)
            {
                return RaiseParseError<SqlCollectionExpression>(columnDefinition.RightValue);
            }
            if (columnDefinition.LeftValue == SqlNoneExpression.Default)
            {
                return RaiseParseError<SqlCollectionExpression>("Expected column definition");
            }

            var column = (ColumnDefinition)columnDefinition.LeftValue;
            ParseColumnConstraints(column);
            columns.Add(column);
            if (_text.PeekChar() != ',')
            {
                break;
            }

            _text.ReadChar();
        } while (!_text.IsEnd());

        return CreateParseResult(columns);
    }

    private Either<SqlCollectionExpression, ParseError> EmptyCollectionResult()
    {
        return new Either<SqlCollectionExpression, ParseError>(new SqlCollectionExpression());
    }

    public Either<ISqlExpression, ParseError> ParseCreateTableStatement()
    {
        if (!TryMatchesKeyword("CREATE", "TABLE"))
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

        var tableColumnsResult = ParseCreateTableColumns();
        if (tableColumnsResult.IsRight)
        {
            return RaiseParseError(tableColumnsResult.RightValue);
        }

        if (tableColumnsResult.LeftValue.SqlType == SqlType.None)
        {
            return RaiseParseError("Expected column definition");
        }

        createTableStatement.Columns = tableColumnsResult.LeftValue.ToList<ColumnDefinition>();

        if (_text.PeekChar() != ')')
        {
            var tableConstraints = ParseWithComma(ParseTableConstraint);
            if (tableConstraints.IsRight)
            {
                return RaiseParseError(tableConstraints.RightValue);
            }
            createTableStatement.Constraints = tableConstraints.LeftValue.Items;
        }

        if (!_text.TryMatch(")"))
        {
            return RaiseParseError("ParseCreateTableStatement Expected )");
        }

        SkipStatementEnd();

        return CreateParseResult(createTableStatement);
    }

    public Either<ISqlExpression, ParseError> ParseExecSpAddExtendedProperty()
    {
        if (!TryMatchesKeyword("EXEC", "SP_AddExtendedProperty"))
        {
            return NoneResult();
        }

        var parameters = ParseWithComma(ParseParameterValueOrAssignValue);
        if (parameters.IsRight)
        {
            return RaiseParseError(parameters.RightValue);
        }
        if (parameters.LeftValue.Items.Count != 8)
        {
            return RaiseParseError("Expected 8 parameters");
        }

        var p = parameters.LeftValue.ToList<SqlParameterValue>();

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

    public Either<ISqlExpression, ParseError> ParseSelectStatement()
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
            if (leftExpr.IsRight)
            {
                return RaiseParseError(leftExpr.RightValue);
            }
            if (leftExpr.LeftValue.SqlType == SqlType.None)
            {
                return RaiseParseError("Expected left expression");
            }

            var operation = _text.ReadSymbols().Word;
            var rightExpr = ParseValue();
            if (rightExpr.IsRight)
            {
                return RaiseParseError(rightExpr.RightValue);
            }
            if (rightExpr.LeftValue.SqlType == SqlType.None)
            {
                return RaiseParseError("Expected right expression");
            }

            selectStatement.Where = new SqlWhereExpression()
            {
                Left = leftExpr.LeftValue,
                Operation = operation,
                Right = rightExpr.LeftValue
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

    public bool Try(Func<Either<ISqlExpression, ParseError>> parseFunc, out Either<ISqlExpression, ParseError> result)
    {
        var localResult = parseFunc();
        if (localResult.IsRight)
        {
            result = localResult;
            return true;
        }

        if (localResult.LeftValue == SqlNoneExpression.Default)
        {
            result = localResult;
            return false;
        }

        result = localResult;
        return true;
    }

    private void MatchString(string expected)
    {
        SkipWhiteSpace();
        _text.Match(expected);
    }

    private Either<ColumnDefinition, ParseError> ParseColumnConstraints(ColumnDefinition column)
    {
        do
        {
            var startPosition = _text.Position; 
            if (TryMatchesKeyword("PRIMARY", "KEY"))
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
                if (identityResult.IsRight)
                {
                    return RaiseParseError<ColumnDefinition>(identityResult.RightValue);
                }
                column.Identity = (SqlIdentity)identityResult.LeftValue;
            }

            if (Try(ParseDefaultValue, out var nonConstraintDefaultValue))
            {
                if (identityResult.IsRight)
                {
                    return RaiseParseError<ColumnDefinition>(identityResult.RightValue);
                }
                column.Constraints.Add((SqlConstraint)nonConstraintDefaultValue.LeftValue);
                continue;
            }

            var constraintStartPosition = _text.Position;
            if (_text.TryMatch(ConstraintKeyword))
            {
                var constraintName = _text.ReadSqlIdentifier();
                if (Try(ParseDefaultValue, out var constraintDefaultValue))
                {
                    if (identityResult.IsRight)
                    {
                        return RaiseParseError<ColumnDefinition>(identityResult.RightValue);
                    }
                    var subConstraint = (SqlConstraint)constraintDefaultValue.LeftValue;
                    subConstraint.ConstraintName = constraintName.Word;
                    column.Constraints.Add(subConstraint);
                    continue;
                }
                _text.Position = constraintStartPosition;
                var columnConstraint = ParseTableConstraint();
                if (columnConstraint.IsRight)
                {
                    return RaiseParseError<ColumnDefinition>(columnConstraint.RightValue);
                }
                if(columnConstraint.LeftValue.SqlType != SqlType.None)
                {
                    var t = (SqlConstraint)columnConstraint.LeftValue;
                    column.Constraints.Add(t);
                }
                return RaiseParseError<ColumnDefinition>("Expect Constraint DEFAULT");
            }

            if (TryMatchesKeyword("NOT", "FOR", "REPLICATION"))
            {
                column.NotForReplication = true;
                continue;
            }

            if (TryMatchesKeyword("NOT", "NULL"))
            {
                column.IsNullable = false;
                continue;
            }

            if (TryMatchesKeyword("NULL"))
            {
                column.IsNullable = true;
                continue;
            }

            break;
        } while (true);

        return ToParseResult(column);
    }

    private Either<ISqlExpression, ParseError> ParseColumnTypeDefinition(TextSpan columnNameSpan)
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

    private Either<ISqlExpression, ParseError> ParseDefaultValue()
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
            return CreateParseResult(new SqlConstraint
            {
                ConstraintName = string.Empty,
                DefaultValue = defaultValue.Word
            });
        }

        var nullValue = _text.PeekIdentifier("NULL");
        if (nullValue.Length > 0)
        {
            _text.ReadIdentifier();
            return CreateParseResult(new SqlConstraint
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
            return CreateParseResult(new SqlConstraint
            {
                ConstraintName = string.Empty,
                DefaultValue = defaultValue.Word,
            });
        }

        defaultValue = _text.ReadNumber();
        return CreateParseResult(new SqlConstraint
        {
            ConstraintName = string.Empty,
            DefaultValue = defaultValue.Word,
        });
    }

    private Either<ISqlExpression, ParseError> ParseIntValue()
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

    private Either<SqlCollectionExpression, ParseError> ParseParenthesesWithComma<T>(Func<Either<T, ParseError>> parseElemFn)
        where T: ISqlExpression 
    {
        if (!_text.TryMatch("("))
        {
            return RaiseParseError<SqlCollectionExpression>("Expected (");
        }

        var elements = ParseWithComma(parseElemFn);
        if (elements.IsRight)
        {
            return RaiseParseError<SqlCollectionExpression>(elements.RightValue);
        }

        if (!_text.TryMatch(")"))
        {
            return RaiseParseError<SqlCollectionExpression>("Expected )");
        }

        return elements;
    }
    
    private Either<SqlCollectionExpression, ParseError> CreateParseResult<T>(IEnumerable<T> result)
    {
        return new Either<SqlCollectionExpression, ParseError>(new SqlCollectionExpression()
        {
            Items = result.Cast<ISqlExpression>().ToList()
        });
    }
    
    private Either<ISqlExpression, ParseError> CreateParseResult<T>(T result)
        where T : ISqlExpression 
    {
        return new Either<ISqlExpression, ParseError>(result);
    }
    
    private Either<T, ParseError> ToParseResult<T>(T result)
    {
        return new Either<T, ParseError>(result);
    }
    
    private Either<ISqlExpression, ParseError> NoneResult()
    {
        return new Either<ISqlExpression, ParseError>(SqlNoneExpression.Default);
    }

    private Either<ISqlExpression, ParseError> ParseTableConstraint()
    {
        var constraintName = string.Empty;
        if (TryMatchKeyword(ConstraintKeyword))
        {
            constraintName = _text.ReadSqlIdentifier().Word;
        }

        var sqlConstraint = new SqlConstraint
        {
            ConstraintName = constraintName
        };
        
        var primaryKeyOrUnique = ParsePrimaryKeyOrUnique();
        if (primaryKeyOrUnique.IsRight)
        {
            return RaiseParseError(primaryKeyOrUnique.RightValue);
        }
        if (primaryKeyOrUnique.IsLeft && primaryKeyOrUnique.LeftValue.SqlType != SqlType.None)
        {
            var subConstraint = (SqlConstraint)primaryKeyOrUnique.LeftValue;
            sqlConstraint.ConstraintType = subConstraint.ConstraintType;
            sqlConstraint.Clustered = subConstraint.Clustered;
            sqlConstraint.Columns = subConstraint.Columns;
        }

        if (TryMatchesKeyword("FOREIGN", "KEY"))
        {
            sqlConstraint.ConstraintType = "FOREIGN KEY";
            var uniqueColumns = ParseParenthesesWithComma(() =>
            {
                var uniqueColumn = _text.ReadSqlIdentifier();
                return CreateParseResult(new SqlConstraintColumn
                {
                    ColumnName = uniqueColumn.Word,
                    Order = string.Empty,
                });
            });
            if (uniqueColumns.IsRight)
            {
                return RaiseParseError(uniqueColumns.RightValue);
            }
            sqlConstraint.Columns = uniqueColumns.LeftValue.ToList<SqlConstraintColumn>();
        }

        if (TryMatchKeyword("WITH"))
        {
            var togglesResult = ParseParenthesesWithComma(ParseWithToggle);
            if (togglesResult.IsRight)
            {
                return RaiseParseError(togglesResult.RightValue);
            }
            sqlConstraint.WithToggles = togglesResult.LeftValue.ToList<SqlToggle>();
        }

        if (TryMatchKeyword("ON"))
        {
            sqlConstraint.On = _text.ReadSqlIdentifier().Word;
        }

        if (Try(ParseIdentity, out var identityResult))
        {
            if (identityResult.IsRight)
            {
                return RaiseParseError(identityResult.RightValue);
            }
            sqlConstraint.Identity = (SqlIdentity)identityResult.LeftValue;
        }

        return CreateParseResult(sqlConstraint);
    }

    private Either<ISqlExpression, ParseError> ParsePrimaryKeyOrUnique()
    {
        var sqlConstraint = new SqlConstraint();
        var primaryKeyOrUniqueToken = Optional(Or(Keywords("PRIMARY", "KEY"), Keywords("UNIQUE")));
        if( primaryKeyOrUniqueToken.SqlType != SqlType.None )
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
        var uniqueColumns = ParseParenthesesWithComma(() =>
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
        if (uniqueColumns.IsRight)
        {
            return RaiseParseError(uniqueColumns.RightValue);
        }
        sqlConstraint.Columns = uniqueColumns.LeftValue.ToList<SqlConstraintColumn>();
        return CreateParseResult(sqlConstraint);
    }

    private ISqlExpression Optional(Func<Either<ISqlExpression, ParseError>> parseFn)
    {
        var result = parseFn();
        if (result.IsLeft && result.LeftValue.SqlType != SqlType.None)
        {
            return result.LeftValue;
        }
        return SqlNoneExpression.Default;
    }

    private Either<ISqlExpression, ParseError> ParseTableName()
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

    private Either<ISqlExpression, ParseError> ParseValue()
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

    private Either<SqlCollectionExpression, ParseError> ParseWithComma<T>(Func<Either<T, ParseError>> parseElemFn)
        where T : ISqlExpression 
    {
        var elements = new List<T>();
        do
        {
            var elem = parseElemFn();
            if (elem is { IsLeft: true, LeftValue.SqlType: SqlType.None })
            {
                break;
            }

            if (elem.IsRight)
            {
                return RaiseParseError<SqlCollectionExpression>(elem.RightValue);
            }
            elements.Add(elem.LeftValue);
            if (_text.PeekChar() != ',')
            {
                break;
            }

            _text.ReadChar();
        } while (!_text.IsEnd());

        return CreateParseResult(elements);
    }

    private Either<ISqlExpression, ParseError> ParseWithToggle()
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

    private Either<ISqlExpression, ParseError> RaiseParseError(ParseError innerError)
    {
        return new Either<ISqlExpression, ParseError>(innerError);
    }

    private Either<ISqlExpression, ParseError> RaiseParseError(string error)
    {
        return new Either<ISqlExpression, ParseError>(new ParseError(error)
        {
            Offset = _text.Position
        });
    }

    private Either<T, ParseError> RaiseParseError<T>(string error)
    {
        return new Either<T, ParseError>(new ParseError(error)
        {
            Offset = _text.Position
        });
    }

    private Either<T, ParseError> RaiseParseError<T>(ParseError innerError)
    {
        return new Either<T, ParseError>(innerError);
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

    private bool TryMatchesKeyword(params string[] keywords)
    {
        SkipWhiteSpace();
        return _text.TryMatchesIgnoreCase(keywords);
    }

    private bool TryMatchKeyword(string expected)
    {
        SkipWhiteSpace();
        return _text.TryMatchIgnoreCaseKeyword(expected);
    }


    private Func<Either<ISqlExpression, ParseError>> Keywords(params string[] keywords)
    {
        return () => ParseKeywords(keywords);
    }

    private Either<ISqlExpression, ParseError> ParseKeywords(params string[] keywords)
    {
        if (TryMatchesKeyword(keywords))
        {
            return CreateParseResult(new SqlToken
            {
                Value= string.Join(" ", keywords) 
            });
        }
        return NoneResult();
    }
    
    private Func<Either<ISqlExpression, ParseError>> Or(params Func<Either<ISqlExpression, ParseError>>[] parseFnList)
    {
        return () =>
        {
            foreach (var parseFn in parseFnList)
            {
                var rc = parseFn();
                if (rc.IsLeft && rc.LeftValue.SqlType != SqlType.None)
                {
                    return rc;
                }

                if (rc.IsRight)
                {
                    return rc;
                }
            }
            return NoneResult();
        };
    }

    private bool TryMatchPrimaryKeyOrUnique(SqlConstraint sqlConstraint)
    {
        if (TryMatchKeyword("UNIQUE"))
        {
            sqlConstraint.ConstraintType = "UNIQUE";
            return true;
        }

        if (TryMatchesKeyword("PRIMARY", "KEY"))
        {
            sqlConstraint.ConstraintType = "PRIMARY KEY";
            return true;
        }

        return false;
    }

    private Either<ISqlExpression, ParseError> ParseParameterValueOrAssignValue()
    {
        var rc1 = ParseParameterValue();
        if (rc1.IsRight)
        {
            return RaiseParseError(rc1.RightValue);
        }
        if (rc1.IsLeft && rc1.LeftValue.SqlType != SqlType.None)
        {
            return rc1;
        }

        var rc2 = ParseParameterAssignValue();
        if (rc2.IsRight)
        {
            return RaiseParseError(rc2.RightValue);
        }

        if (rc2.IsLeft && rc2.LeftValue.SqlType != SqlType.None)
        {
            return rc2;
        }

        return NoneResult();
    }

    private Either<ISqlExpression, ParseError> ParseParameterValue()
    {
        SkipWhiteSpace();
        var startPosition = _text.Position;
        var valueResult = ParseValue();
        if (valueResult.IsRight)
        {
            _text.Position = startPosition;
            return RaiseParseError(valueResult.RightValue);
        }
        if (valueResult.LeftValue.SqlType == SqlType.None)
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
            Value = ((ISqlValue)valueResult.LeftValue).Value
        });
    }

    private Either<ISqlExpression, ParseError> ParseParameterAssignValue()
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

    private Either<ISqlExpression, ParseError> ParseIdentity()
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

// public class ParseResult
// {
//     public ISqlExpression Result { get; set; } = SqlNoneExpression.Default;
//     public ParseError Error { get; set; } = ParseError.Empty;
// }