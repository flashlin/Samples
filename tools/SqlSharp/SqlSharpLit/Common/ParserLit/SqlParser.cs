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
        if (TryStart(ParseCreateTableStatement, out var createTableResult))
        {
            return createTableResult;
        }

        if (TryStart(ParseSelectStatement, out var selectResult))
        {
            return selectResult;
        }

        if (TryStart(ParseExecSpAddExtendedProperty, out var execSpAddExtendedPropertyResult))
        {
            return execSpAddExtendedPropertyResult;
        }
        return new Either<ISqlExpression, ParseError>(new ParseError("Unknown statement"));
    }
    

    public Either<ColumnDefinition[], ParseError> ParseCreateTableColumns()
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
                return RaiseParseError<ColumnDefinition>($"Expected column name, but got {_text.PeekWord().Word}");
            }

            var columnDefinition = ParseColumnTypeDefinition(item);
            if (columnDefinition.IsRight)
            {
                return RaiseParseError<ColumnDefinition>(columnDefinition.RightValue);
            }

            if (columnDefinition.LeftValue.Length == 0)
            {
                return RaiseParseError<ColumnDefinition>("Expected column definition");
            }

            var column = columnDefinition.LeftValue.First();
            ParseColumnConstraints(column);
            columns.Add(column);
            if (_text.PeekChar() != ',')
            {
                break;
            }
            _text.ReadChar();
        } while (!_text.IsEnd());

        return ParseResult<ColumnDefinition>(columns);
    }

    public Either<ISqlExpression[], ParseError> ParseCreateTableStatement()
    {
        if (!TryMatchesKeyword("CREATE", "TABLE"))
        {
            return ParseResult<ISqlExpression>();
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
        if( tableColumnsResult.LeftValue.Length == 0)
        {
            return RaiseParseError("Expected column definition");
        }

        createTableStatement.Columns = tableColumnsResult.LeftValue.ToList();

        if (_text.PeekChar() != ')')
        {
            var tableConstraints = ParseWithComma(ParseTableConstraint);
            if (tableConstraints.IsRight)
            {
                return RaiseParseError(tableConstraints.RightValue);
            }

            createTableStatement.Constraints = tableConstraints.LeftValue.First();
        }

        if (!_text.TryMatch(")"))
        {
            return RaiseParseError("ParseCreateTableStatement Expected )");
        }

        SkipStatementEnd();

        return ParseResult<ISqlExpression>(createTableStatement);
    }

    public Either<ISqlExpression[], ParseError> ParseExecSpAddExtendedProperty()
    {
        if (!TryMatchesKeyword("EXEC", "SP_AddExtendedProperty"))
        {
            return CreateStartParseError("Expected EXEC SP_AddExtendedProperty");
        }

        var parameters = ParseWithComma(() =>
        {
            var parameter = ParseParameterValueOrAssignValue();
            return parameter;
        });
        if (parameters.IsRight)
        {
            return RaiseParseError<ISqlExpression>(parameters.RightValue);
        }
        if(parameters.LeftValue.First().Count != 8)
        {
            return RaiseParseError("Expected 8 parameters");
        }
        var p = parameters.LeftValue.First();

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
        return ParseResult<ISqlExpression>(sqlSpAddExtendedProperty);
    }

    public Either<ISqlExpression[], ParseError> ParseSelectStatement()
    {
        if (!TryMatchKeyword("SELECT"))
        {
            return CreateStartParseError(
                $"Expected SELECT, but got {_text.PreviousWord().Word} {_text.PeekWord().Word}");
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
            if (leftExpr.LeftValue.Length == 0)
            {
                return RaiseParseError<ISqlExpression>("Expected left expression");
            }

            var operation = _text.ReadSymbols().Word;
            var rightExpr = ParseValue();
            if (rightExpr.IsRight)
            {
                return RaiseParseError(rightExpr.RightValue);
            }
            if (rightExpr.LeftValue.Length == 0)
            {
                return RaiseParseError<ISqlExpression>("Expected right expression");
            }

            selectStatement.Where = new SqlWhereExpression()
            {
                Left = leftExpr.LeftValue.First(),
                Operation = operation,
                Right = rightExpr.LeftValue.First()
            };
        }

        SkipStatementEnd();
        return ParseResult<ISqlExpression>(selectStatement);
    }

    public void SkipStatementEnd()
    {
        var ch = _text.PeekChar();
        if (ch == ';')
        {
            _text.ReadChar();
        }
    }

    public bool Try<T>(Func<Either<T[], ParseError>> parseFunc, out Either<T[], ParseError> result)
    {
        var localResult = parseFunc();
        if (localResult.IsRight)
        {
            result = localResult;
            return true;
        }

        if (localResult.LeftValue.Length == 0)
        {
            result = localResult;
            return false;
        }

        result = localResult;
        return true;
    }

    private Either<T, ParseError> CreateParseError<T>(string message)
    {
        return new Either<T, ParseError>(new ParseError(message)
        {
            Offset = _text.Position
        });
    }

    private Either<ISqlExpression[], ParseError> CreateStartParseError(string message)
    {
        return new Either<ISqlExpression[], ParseError>(new ParseError(message)
        {
            Offset = _text.Position,
            IsStart = true
        });
    }

    private void MatchString(string expected)
    {
        SkipWhiteSpace();
        _text.Match(expected);
    }

    private ParseError ParseColumnConstraints(ColumnDefinition column)
    {
        do
        {
            if (TryMatchesKeyword("PRIMARY", "KEY"))
            {
                column.IsPrimaryKey = true;
                continue;
            }

            if (TryParseSqlIdentity(column, out var identityResult))
            {
                if (identityResult.IsRight)
                {
                    return identityResult.RightValue;
                }

                continue;
            }

            if (Try(ParseDefaultValue, out var nonConstraintDefaultValue))
            {
                if (identityResult.IsRight)
                {
                    return identityResult.RightValue;
                }

                column.Constraints.Add(nonConstraintDefaultValue.LeftValue.First());
                continue;
            }

            if (_text.TryMatch(ConstraintKeyword))
            {
                var constraintName = _text.ReadSqlIdentifier();
                if (Try(ParseDefaultValue, out var constraintDefaultValue))
                {
                    if (identityResult.IsRight)
                    {
                        return identityResult.RightValue;
                    }

                    column.Constraints.Add(new SqlConstraintDefault
                    {
                        ConstraintName = constraintName.Word,
                        Value = constraintDefaultValue.LeftValue.First().Value
                    });
                    continue;
                }

                return new ParseError("Expect Constraint DEFAULT");
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

        return ParseError.Empty;
    }

    private Either<ColumnDefinition[], ParseError> ParseColumnTypeDefinition(TextSpan columnNameSpan)
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
                return ParseResult(column);
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
                return RaiseParseError<ColumnDefinition>("Expected )");
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

        return ParseResult(column);
    }

    private Either<SqlConstraintDefault[], ParseError> ParseDefaultValue()
    {
        if (!TryMatchKeyword("DEFAULT"))
        {
            return ParseResult<SqlConstraintDefault>();
        }

        TextSpan defaultValue;
        if (_text.TryMatch("("))
        {
            defaultValue = _text.ReadUntilRightParenthesis();
            _text.Match(")");
            return ParseResult(new SqlConstraintDefault
            {
                ConstraintName = "[DEFAULT]",
                Value = defaultValue.Word
            });
        }

        var nullValue = _text.PeekIdentifier("NULL");
        if (nullValue.Length > 0)
        {
            _text.ReadIdentifier();
            return ParseResult(new SqlConstraintDefault
            {
                ConstraintName = "[DEFAULT]",
                Value = nullValue.Word
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
            return ParseResult(new SqlConstraintDefault
            {
                ConstraintName = "[DEFAULT]",
                Value = defaultValue.Word,
            });
        }

        defaultValue = _text.ReadNumber();
        return ParseResult(new SqlConstraintDefault
        {
            ConstraintName = "[DEFAULT]",
            Value = defaultValue.Word,
        });
    }

    private Either<ISqlExpression[], ParseError> ParseIntValue()
    {
        if (_text.Try(_text.ReadNumber, out var number))
        {
            return ParseResult<ISqlExpression>(new SqlIntValueExpression
            {
                Value = number.Word
            });
        }
        return ParseResult<ISqlExpression>();
    }

    private Either<List<T>[], ParseError> ParseParenthesesWithComma<T>(Func<Either<T[], ParseError>> parseElemFn)
    {
        if (!_text.TryMatch("("))
        {
            return RaiseParseError<List<T>>("Expected (");
        }

        var elements = ParseWithComma(parseElemFn);
        if (elements.IsRight)
        {
            return RaiseParseError<List<T>>(elements.RightValue);
        }

        if (!_text.TryMatch(")"))
        {
            return RaiseParseError<List<T>>("Expected )");
        }

        return elements;
    }

    private Either<T[], ParseError> ParseResult<T>(IEnumerable<T> result)
    {
        return new Either<T[], ParseError>(result.ToArray());
    }

    private Either<T[], ParseError> ParseResult<T>(T result)
    {
        return new Either<T[], ParseError>([result]);
    }

    private Either<T[], ParseError> ParseResult<T>()
    {
        return new Either<T[], ParseError>([]);
    }

    private Either<T2, ParseError> ParseResult<T1, T2>(Either<T1, ParseError> result, Func<T1, T2> toResult)
    {
        if (result.IsLeft)
        {
            return new Either<T2, ParseError>(toResult(result.LeftValue));
        }

        return new Either<T2, ParseError>(result.RightValue);
    }

    private Either<ISqlExpression[], ParseError> ParseTableConstraint()
    {
        var constraintName = "DEFAULT";
        if (TryMatchKeyword(ConstraintKeyword))
        {
            constraintName = _text.ReadSqlIdentifier().Word;
        }

        var sqlConstraint = new SqlConstraint
        {
            ConstraintName = constraintName
        };

        if (TryMatchPrimaryKeyOrUnique(sqlConstraint))
        {
            if (TryMatchKeyword("CLUSTERED"))
            {
                sqlConstraint.Clustered = "CLUSTERED";
            }
            else if (TryMatchesKeyword("NONCLUSTERED"))
            {
                sqlConstraint.Clustered = "NONCLUSTERED";
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

                return ParseResult(new SqlConstraintColumn
                {
                    ColumnName = columnName.Word,
                    Order = order,
                });
            });
            if (uniqueColumns.IsRight)
            {
                return RaiseParseError(uniqueColumns.RightValue);
            }
            sqlConstraint.Columns = uniqueColumns.LeftValue.First();
        }

        if (TryMatchesKeyword("FOREIGN", "KEY"))
        {
            sqlConstraint.ConstraintType = "FOREIGN KEY";
            var uniqueColumns = ParseParenthesesWithComma(() =>
            {
                var uniqueColumn = _text.ReadSqlIdentifier();
                return ParseResult(new SqlConstraintColumn
                {
                    ColumnName = uniqueColumn.Word,
                    Order = string.Empty,
                });
            });
            if (uniqueColumns.IsRight)
            {
                return RaiseParseError(uniqueColumns.RightValue);
            }

            sqlConstraint.Columns = uniqueColumns.LeftValue.First();
        }

        if (TryMatchKeyword("WITH"))
        {
            var togglesResult = ParseParenthesesWithComma(ParseWithToggle);
            if (togglesResult.IsRight)
            {
                return RaiseParseError(togglesResult.RightValue);
            }
            sqlConstraint.WithToggles = togglesResult.LeftValue.First();
        }

        if (TryMatchKeyword("ON"))
        {
            sqlConstraint.On = _text.ReadSqlIdentifier().Word;
        }

        return ParseResult<ISqlExpression>(sqlConstraint);
    }

    private Either<ISqlExpression[], ParseError> ParseTableName()
    {
        if (_text.Try(_text.ReadIdentifier, out var fieldName))
        {
            return ParseResult<ISqlExpression>(new SqlFieldExpression()
            {
                FieldName = fieldName.Word
            });
        }
        return ParseResult<ISqlExpression>();
    }

    private Either<ISqlExpression[], ParseError> ParseValue()
    {
        if (Try(ParseIntValue, out var number))
        {
            return number;
        }
        
        if (_text.Try(_text.ReadSqlQuotedString, out var quotedString))
        {
            return ParseResult<ISqlExpression>(new SqlStringValue
            {
                Value = quotedString.Word
            });
        }

        if (Try(ParseTableName, out var tableName))
        {
            return tableName;
        }
        
        return ParseResult<ISqlExpression>();
    }

    private Either<List<T>[], ParseError> ParseWithComma<T>(Func<Either<T[], ParseError>> parseElemFn)
    {
        var elements = new List<T>();
        do
        {
            var elem = parseElemFn();
            if (elem.IsLeft && elem.LeftValue.Length == 0)
            {
                break;
            }

            if (elem.IsRight)
            {
                return RaiseParseError<List<T>>(elem.RightValue);
            }

            elements.Add(elem.LeftValue.First());
            if (_text.PeekChar() != ',')
            {
                break;
            }

            _text.ReadChar();
        } while (!_text.IsEnd());

        return ParseResult(elements);
    }

    private Either<SqlWithToggle[], ParseError> ParseWithToggle()
    {
        var toggle = new SqlWithToggle
        {
            ToggleName = _text.ReadSqlIdentifier().Word
        };
        _text.Match("=");

        if (_text.Try(_text.ReadNumber, out var number))
        {
            toggle.Value = number.Word;
            return ParseResult(toggle);
        }

        toggle.Value = _text.ReadSqlIdentifier().Word;
        return ParseResult(toggle);
    }

    private bool PeekMatchSymbol(string symbol)
    {
        SkipWhiteSpace();
        return _text.PeekMatchSymbol(symbol);
    }

    private Either<ISqlExpression[], ParseError> RaiseParseError(ParseError innerError)
    {
        return new Either<ISqlExpression[], ParseError>(innerError);
    }

    private Either<ISqlExpression[], ParseError> RaiseParseError(string error)
    {
        return new Either<ISqlExpression[], ParseError>(new ParseError(error)
        {
            IsStart = false,
            Offset = _text.Position
        });
    }

    private Either<T[], ParseError> RaiseParseError<T>(string error)
    {
        return new Either<T[], ParseError>(new ParseError(error)
        {
            IsStart = false,
            Offset = _text.Position
        });
    }

    private Either<T[], ParseError> RaiseParseError<T>(ParseError innerError)
    {
        return new Either<T[], ParseError>(innerError);
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
    
    private Either<SqlParameterValue[], ParseError> ParseParameterValueOrAssignValue()
    {
        var rc1 = ParseParameterValue();
        if (rc1.IsRight)
        {
            return RaiseParseError<SqlParameterValue>(rc1.RightValue);
        }
        if (rc1.IsLeft && rc1.LeftValue.Length != 0)
        {
            return rc1;
        }
        var rc2 = ParseParameterAssignValue();
        if (rc2.IsRight)
        {
            return RaiseParseError<SqlParameterValue>(rc2.RightValue);
        }
        if (rc2.IsLeft && rc2.LeftValue.Length != 0)
        {
            return rc2;
        }
        return ParseResult<SqlParameterValue>();
    }
    
    private Either<SqlParameterValue[], ParseError> ParseParameterValue()
    {
        SkipWhiteSpace();
        var startPosition = _text.Position;
        var valueResult = ParseValue();
        if (valueResult.IsRight)
        {
            _text.Position = startPosition;
            return RaiseParseError<SqlParameterValue>(valueResult.RightValue);
        }
        if(valueResult.LeftValue.Length == 0)
        {
            return ParseResult<SqlParameterValue>();
        }

        if (_text.Peek(_text.ReadSymbols).Word == "=")
        {
            _text.Position = startPosition;
            return ParseResult<SqlParameterValue>();
        }
        
        return ParseResult(new SqlParameterValue
        {
            Name = string.Empty,
            Value = ((ISqlValue)valueResult.LeftValue.First()).Value
        });
    }

    private Either<SqlParameterValue[], ParseError> ParseParameterAssignValue()
    {
        SkipWhiteSpace();
        if (!_text.Try(_text.ReadSqlIdentifier, out var name))
        {
            return ParseResult<SqlParameterValue>();
        }

        if (!_text.TryMatch("="))
        {
            return RaiseParseError<SqlParameterValue>("Expected =");
        }

        if (!_text.Try(_text.ReadSqlQuotedString, out var nameValue))
        {
            return RaiseParseError<SqlParameterValue>($"Expected @name value, but got {_text.PreviousWord().Word}");
        }

        return ParseResult(new SqlParameterValue
        {
            Name = name.Word,
            Value = nameValue.Word
        });
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

    private bool TryStart(Func<Either<ISqlExpression[], ParseError>> parseFunc,
        out Either<ISqlExpression, ParseError> result)
    {
        if (Try(parseFunc, out var parseResult))
        {
            if (parseResult is { IsRight: true, RightValue.IsStart: true })
            {
                result = new Either<ISqlExpression, ParseError>(parseResult.RightValue);
                return false;
            }
            if (parseResult.IsRight)
            {
                result = new Either<ISqlExpression, ParseError>(parseResult.RightValue);
                return true;
            }
            result = new Either<ISqlExpression, ParseError>(parseResult.LeftValue.First());
            return true;
        }
        result = new Either<ISqlExpression, ParseError>(ParseError.Empty);
        return false;
    }
}

public class SqlStringValue : ISqlValue, ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.String;
    public string Value { get; set; } = string.Empty;
    public string ToSql()
    {
        return $"{Value}";
    }
}