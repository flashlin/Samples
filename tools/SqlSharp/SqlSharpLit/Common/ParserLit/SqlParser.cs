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

        return CreateStartParseError("Unknown statement");
    }

    public Either<List<ColumnDefinition>, ParseError> ParseCreateTableColumns()
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
                return new Either<List<ColumnDefinition>, ParseError>(
                    new ParseError($"Expected column name, but got {_text.PeekWord().Word}")
                    {
                        Offset = _text.Position
                    });
            }

            var columnDefinition = ParseColumnTypeDefinition(item);
            if (columnDefinition.IsRight)
            {
                return new Either<List<ColumnDefinition>, ParseError>(columnDefinition.RightValue);
            }

            _text.SkipSqlComment();

            var column = columnDefinition.LeftValue;
            if (ParseColumnConstraints(column) is var error && error != ParseError.Empty)
            {
                return new Either<List<ColumnDefinition>, ParseError>(error);
            }

            columns.Add(column);
            if (_text.PeekChar() != ',')
            {
                break;
            }

            _text.ReadChar();
        } while (!_text.IsEnd());

        return new Either<List<ColumnDefinition>, ParseError>(columns);
    }

    public Either<ISqlExpression, ParseError> ParseCreateTableStatement()
    {
        if (!TryMatchesKeyword("CREATE", "TABLE"))
        {
            return CreateStartParseError(
                $"Expected CREATE TABLE, but got {_text.PreviousWord().Word} {_text.PeekWord().Word}");
        }

        var tableName = _text.ReadSqlIdentifier();
        if (!_text.TryMatch("("))
        {
            return CreateParseError("Expected (");
        }

        var createTableStatement = new CreateTableStatement()
        {
            TableName = tableName.Word,
        };

        var rc = ParseCreateTableColumns();
        if (rc.IsRight)
        {
            return CreateParseError(rc.RightValue.Message);
        }

        createTableStatement.Columns = rc.LeftValue;

        if (_text.PeekChar() != ')')
        {
            var tableConstraints = ParseWithComma(ParseTableConstraint);
            if (tableConstraints.IsRight)
            {
                return RaiseParseError(tableConstraints.RightValue);
            }
            createTableStatement.Constraints = tableConstraints.LeftValue;
        }

        if (!_text.TryMatch(")"))
        {
            return CreateParseError("ParseCreateTableStatement Expected )");
        }

        SkipStatementEnd();

        return new Either<ISqlExpression, ParseError>(createTableStatement);
    }

    public Either<ISqlExpression, ParseError> ParseExecSpAddExtendedProperty()
    {
        if (!TryMatchesKeyword("EXEC", "SP_AddExtendedProperty"))
        {
            return CreateStartParseError("Expected EXEC SP_AddExtendedProperty");
        }

        if (!TryMatchParameterAssignValue("@name", out var nameParameter))
        {
            return CreateStartParseError(nameParameter.RightValue.Message);
        }

        MatchString(",");

        if (!TryMatchParameterAssignValue("@value", out var valueParameter))
        {
            return CreateStartParseError(valueParameter.RightValue.Message);
        }

        MatchString(",");

        if (!TryMatchParameterAssignValue("@level0type", out var level0TypeParameter))
        {
            return CreateStartParseError(level0TypeParameter.RightValue.Message);
        }

        MatchString(",");

        if (!TryMatchParameterAssignValue("@level0name", out var level0NameParameter))
        {
            return CreateStartParseError(level0NameParameter.RightValue.Message);
        }

        MatchString(",");

        if (!TryMatchParameterAssignValue("@level1type", out var level1TypeParameter))
        {
            return CreateStartParseError(level1TypeParameter.RightValue.Message);
        }

        MatchString(",");

        if (!TryMatchParameterAssignValue("@level1name", out var level1NameParameter))
        {
            return CreateStartParseError(level1NameParameter.RightValue.Message);
        }

        MatchString(",");

        if (!TryMatchParameterAssignValue("@level2type", out var level2TypeParameter))
        {
            return CreateStartParseError(level2TypeParameter.RightValue.Message);
        }

        MatchString(",");

        if (!TryMatchParameterAssignValue("@level2name", out var level2NameParameter))
        {
            return CreateStartParseError(level2NameParameter.RightValue.Message);
        }

        SkipStatementEnd();
        var sqlSpAddExtendedProperty = new SqlSpAddExtendedProperty
        {
            Name = nameParameter.LeftValue.Value,
            Value = valueParameter.LeftValue.Value,
            Level0Type = level0TypeParameter.LeftValue.Value,
            Level0Name = level0NameParameter.LeftValue.Value,
            Level1Type = level1TypeParameter.LeftValue.Value,
            Level1Name = level1NameParameter.LeftValue.Value,
            Level2Type = level2TypeParameter.LeftValue.Value,
            Level2Name = level2NameParameter.LeftValue.Value
        };
        return new Either<ISqlExpression, ParseError>(sqlSpAddExtendedProperty);
    }

    public Either<ISqlExpression, ParseError> ParseSelectStatement()
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
            if (leftExpr.IsRight)
            {
                return CreateParseError(leftExpr.RightValue.Message);
            }

            var operation = _text.ReadSymbols().Word;
            var rightExpr = ParseValue();
            if (rightExpr.IsRight)
            {
                return CreateParseError(rightExpr.RightValue.Message);
            }

            selectStatement.Where = new SqlWhereExpression()
            {
                Left = leftExpr.LeftValue,
                Operation = operation,
                Right = rightExpr.LeftValue
            };
        }

        SkipStatementEnd();
        return new Either<ISqlExpression, ParseError>(selectStatement);
    }

    public void SkipStatementEnd()
    {
        var ch = _text.PeekChar();
        if (ch == ';')
        {
            _text.ReadChar();
        }
    }

    public bool Try<T>(Func<Either<T, ParseError>> parseFunc, out Either<T, ParseError> result)
    {
        var localResult = new Either<T, ParseError>(new ParseError("Unknown"));
        var rc = parseFunc();
        var success = rc.Match(left =>
            {
                localResult = new Either<T, ParseError>(left);
                return true;
            },
            right =>
            {
                localResult = new Either<T, ParseError>(right);
                return false;
            });
        result = localResult;
        return success;
    }

    private Either<ISqlExpression, ParseError> CreateParseError(string message)
    {
        return CreateParseError<ISqlExpression>(message);
    }

    private Either<T, ParseError> CreateParseError<T>(string message)
    {
        return new Either<T, ParseError>(new ParseError(message)
        {
            Offset = _text.Position
        });
    }

    private Either<ISqlExpression, ParseError> CreateStartParseError(string message)
    {
        return new Either<ISqlExpression, ParseError>(new ParseError(message)
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

                column.Constraints.Add(new SqlConstraintDefault
                {
                    ConstraintName = "[DEFAULT]",
                    Value = nonConstraintDefaultValue.LeftValue.Word
                });
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
                        Value = constraintDefaultValue.LeftValue.Word
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

    private Either<ColumnDefinition, ParseError> ParseColumnTypeDefinition(TextSpan columnNameSpan)
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
                return new Either<ColumnDefinition, ParseError>(column);
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
                return new Either<ColumnDefinition, ParseError>(
                    new ParseError("Expected )")
                    {
                        Offset = _text.Position
                    });
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

        return new Either<ColumnDefinition, ParseError>(column);
    }

    private Either<TextSpan, ParseError> ParseDefaultValue()
    {
        if (!TryMatchKeyword("DEFAULT"))
        {
            return CreateParseError<TextSpan>("Expected DEFAULT");
        }

        TextSpan defaultValue;
        if (_text.TryMatch("("))
        {
            defaultValue = _text.ReadUntilRightParenthesis();
            _text.Match(")");
            return new Either<TextSpan, ParseError>(defaultValue);
        }

        var nullValue = _text.PeekIdentifier("NULL");
        if (nullValue.Length > 0)
        {
            _text.ReadIdentifier();
            return new Either<TextSpan, ParseError>(nullValue);
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
            return new Either<TextSpan, ParseError>(defaultValue);
        }

        defaultValue = _text.ReadNumber();
        return new Either<TextSpan, ParseError>(defaultValue);
    }

    private Either<ISqlExpression, ParseError> ParseIntValue()
    {
        if (_text.Try(_text.ReadNumber, out var number))
        {
            return new Either<ISqlExpression, ParseError>(new SqlIntValueExpression
            {
                Value = int.Parse(number.Word)
            });
        }

        return new Either<ISqlExpression, ParseError>(new ParseError("Expected Int"));
    }

    private Either<List<T>, ParseError> ParseParenthesesWithComma<T>(Func<Either<T, ParseError>> parseElemFn)
    {
        if (!_text.TryMatch("("))
        {
            return CreateParseError<List<T>>("Expected (");
        }

        var elements = ParseWithComma(parseElemFn);
        if (elements.IsRight)
        {
            return RaiseParseError<List<T>>(elements.RightValue);
        }

        if (!_text.TryMatch(")"))
        {
            return CreateParseError<List<T>>("Expected )");
        }

        return elements;
    }

    private Either<T, ParseError> ParseResult<T>(T result)
    {
        return new Either<T, ParseError>(result);
    }

    private Either<T2, ParseError> ParseResult<T1, T2>(Either<T1, ParseError> result, Func<T1, T2> toResult)
    {
        if (result.IsLeft)
        {
            return new Either<T2, ParseError>(toResult(result.LeftValue));
        }

        return new Either<T2, ParseError>(result.RightValue);
    }

    private Either<ISqlExpression, ParseError> ParseTableConstraint()
    {
        // if (!TryMatchKeyword(ConstraintKeyword))
        // {
        //     return CreateParseError("Expected CONSTRAINT");
        // }
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

            sqlConstraint.Columns = uniqueColumns.LeftValue;
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

            sqlConstraint.Columns = uniqueColumns.LeftValue;
        }

        if (TryMatchKeyword("WITH"))
        {
            var togglesResult = ParseParenthesesWithComma(ParseWithToggle);
            if (togglesResult.IsRight)
            {
                return RaiseParseError(togglesResult.RightValue);
            }

            sqlConstraint.WithToggles = togglesResult.LeftValue;
        }

        if (TryMatchKeyword("ON"))
        {
            sqlConstraint.On = _text.ReadSqlIdentifier().Word;
        }

        return new Either<ISqlExpression, ParseError>(sqlConstraint);
    }

    private Either<ISqlExpression, ParseError> ParseTableName()
    {
        if (_text.Try(_text.ReadIdentifier, out var fieldName))
        {
            return new Either<ISqlExpression, ParseError>(new SqlFieldExpression()
            {
                FieldName = fieldName.Word
            });
        }

        return new Either<ISqlExpression, ParseError>(new ParseError("Expected field name"));
    }

    private Either<ISqlExpression, ParseError> ParseValue()
    {
        if (Try(ParseIntValue, out var number))
        {
            return number;
        }

        if (Try(ParseTableName, out var tableName))
        {
            return tableName;
        }

        return CreateParseError("Expected Int or Field");
    }

    private Either<List<T>, ParseError> ParseWithComma<T>(Func<Either<T, ParseError>> parseElemFn)
    {
        var elements = new List<T>();
        do
        {
            var elem = parseElemFn();
            if (elem.IsRight)
            {
                if (elem.RightValue == ParseError.Empty)
                {
                    break;
                }

                return new Either<List<T>, ParseError>(elem.RightValue);
            }

            elements.Add(elem.LeftValue);
            if (_text.PeekChar() != ',')
            {
                break;
            }

            _text.ReadChar();
        } while (!_text.IsEnd());

        return new Either<List<T>, ParseError>(elements);
    }

    private Either<SqlWithToggle, ParseError> ParseWithToggle()
    {
        var toggle = new SqlWithToggle
        {
            ToggleName = _text.ReadSqlIdentifier().Word
        };
        _text.Match("=");

        if (_text.Try(_text.ReadNumber, out var number))
        {
            toggle.Value = number.Word;
            return new Either<SqlWithToggle, ParseError>(toggle);
        }

        toggle.Value = _text.ReadSqlIdentifier().Word;
        return new Either<SqlWithToggle, ParseError>(toggle);
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

    private bool TryMatchParameterAssignValue(string parameterName, out Either<SqlParameterValue, ParseError> result)
    {
        var startPosition = _text.Position;
        if (!TryParameterAssignValue(out result))
        {
            return false;
        }

        if (!string.Equals(result.LeftValue.Name, parameterName, StringComparison.OrdinalIgnoreCase))
        {
            _text.Position = startPosition;
            result = new Either<SqlParameterValue, ParseError>(
                new ParseError($"Expected {parameterName}, but got {result.LeftValue.Name}")
                {
                    Offset = startPosition
                });
            return false;
        }

        return true;
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

    private bool TryParameterAssignValue(out Either<SqlParameterValue, ParseError> result)
    {
        SkipWhiteSpace();
        if (!_text.Try(_text.ReadSqlIdentifier, out var name))
        {
            result = new Either<SqlParameterValue, ParseError>(
                new ParseError($"Expected @name, but got {_text.PreviousWord().Word}"));
            return false;
        }

        _text.Match("=");

        if (!_text.Try(_text.ReadSqlQuotedString, out var nameValue))
        {
            result = new Either<SqlParameterValue, ParseError>(
                new ParseError($"Expected @name value, but got {_text.PreviousWord().Word}"));
            return false;
        }

        result = new Either<SqlParameterValue, ParseError>(new SqlParameterValue
        {
            Name = name.Word,
            Value = nameValue.Word
        });
        return true;
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

    private bool TryStart(Func<Either<ISqlExpression, ParseError>> parseFunc,
        out Either<ISqlExpression, ParseError> result)
    {
        if (Try(parseFunc, out var parseResult))
        {
            result = parseResult;
            return true;
        }

        var error = parseResult.RightValue;
        if (!error.IsStart)
        {
            result = new Either<ISqlExpression, ParseError>(error);
            return true;
        }

        result = new Either<ISqlExpression, ParseError>(error);
        return false;
    }
}