using T1.Standard.DesignPatterns;

namespace SqlSharpLit.Common.ParserLit;

//CONSTRAINT [PK_AcceptedBets] PRIMARY KEY CLUSTERED 
//(
//   [MatchResultID] ASC
//) WITH (PAD_INDEX  = OFF, STATISTICS_NORECOMPUTE  = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS  = ON, ALLOW_PAGE_LOCKS  = ON) ON [PRIMARY]
public class SqlConstraint
{
    public string ConstraintName { get; set; } = string.Empty;
    public string ConstraintType { get; set; } = string.Empty;
    public string Clustered { get; set; } = string.Empty;
    public List<SqlColumnConstraint> Columns { get; set; } = [];
    public List<SqlWithToggle> WithToggles { get; set; } = [];
    public string On { get; set; } = string.Empty;
}

public class SqlColumnConstraint
{
    public string ColumnName { get; set; } = string.Empty;
    public string Order { get; set; } = "ASC";
}

public class SqlWithToggle
{
    public string ToggleName { get; set; } = string.Empty;
    public string Value { get; set; } = string.Empty;
}

public class SqlParser
{
    private const string ConstraintKeyword = "CONSTRAINT";
    private readonly StringParser _text;

    public SqlParser(string text)
    {
        _text = new StringParser(text);
    }

    public Either<ISqlExpression, ParseError> Parse()
    {
        if (Try(ParseCreateTableStatement, out var sqlCreateTableExpr, out var error))
        {
            return new Either<ISqlExpression, ParseError>(sqlCreateTableExpr);
        }

        if (!error.IsStart)
        {
            return new Either<ISqlExpression, ParseError>(error);
        }

        if (Try(ParseSelectStatement, out var sqlSelectExpr, out error))
        {
            return new Either<ISqlExpression, ParseError>(sqlSelectExpr);
        }

        if (!error.IsStart)
        {
            return new Either<ISqlExpression, ParseError>(error);
        }

        return CreateStartParseError("Unknown statement");
    }

    public Either<List<ColumnDefinition>, ParseError> ParseCreateTableColumns()
    {
        var columns = new List<ColumnDefinition>();
        do
        {
            if (_text.IsPeekIdentifier(ConstraintKeyword))
            {
                break;
            }

            var item = _text.ReadSqlIdentifier();
            if (item.Length == 0)
            {
                return new Either<List<ColumnDefinition>, ParseError>(
                    new ParseError($"Expected column name, but got {_text.PeekKeyword().Word}")
                    {
                        Offset = _text.Position
                    });
            }

            var column = ParseDataDeColumnDefinition(item);
            column.Identity = ParseSqlIdentity();

            ParseColumnConstraints(column);

            columns.Add(column);
            if (_text.PeekChar() != ',')
            {
                break;
            }

            _text.NextChar();
        } while (!_text.IsEnd());

        return new Either<List<ColumnDefinition>, ParseError>(columns);
    }

    private void ParseColumnConstraints(ColumnDefinition column)
    {
        do
        {
            if (_text.TryMatch(ConstraintKeyword))
            {
                var constraintName = _text.ReadSqlIdentifier();
                if (_text.TryMatch("DEFAULT"))
                {
                    _text.Match("(");
                    var defaultValue = _text.ReadUntilRightParenthesis();
                    _text.Match(")");
                    column.Constraints.Add(new SqlConstraintDefault
                    {
                        ConstraintName = constraintName.Word,
                        Value = defaultValue.Word
                    });
                    continue;
                }
                throw new ParseError("Expect Constraint DEFAULT");
            }
            if (_text.TryMatches("NOT", "FOR", "REPLICATION"))
            {
                column.NotForReplication = true;
                continue;
            }
            if (_text.TryMatches("NOT", "NULL"))
            {
                column.IsNullable = false;
                continue;
            }
            if (_text.TryMatches("NULL"))
            {
                column.IsNullable = true;
                continue;
            }
            break;
        } while (true);
    }

    public Either<ISqlExpression, ParseError> ParseCreateTableStatement()
    {
        if (!(_text.TryMatchKeyword("CREATE") && _text.TryMatchKeyword("TABLE")))
        {
            return CreateStartParseError(
                $"Expected CREATE TABLE, but got {_text.PreviousWord().Word} {_text.PeekKeyword().Word}");
        }

        var tableName = _text.ReadUntil(c => char.IsWhiteSpace(c) || c == '(');
        _text.Match("(");

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

        var constraint = ParseTableConstraint();
        if (constraint.IsRight)
        {
            return CreateParseError(constraint.RightValue.Message);
        }

        if (constraint is { IsLeft: true, Left: not null })
        {
            createTableStatement.Constraints.Add(constraint.Left);
        }

        return new Either<ISqlExpression, ParseError>(createTableStatement);
    }

    private Either<SqlConstraint?, ParseError> ParseTableConstraint()
    {
        if (!_text.TryMatch(ConstraintKeyword))
        {
            return new Either<SqlConstraint?, ParseError>(default(SqlConstraint));
        }

        var sqlConstraint = new SqlConstraint
        {
            ConstraintName = _text.ReadSqlIdentifier().Word
        };
        var constraintType = _text.ReadIdentifier().Word;
        if (constraintType.ToUpper() == "PRIMARY")
        {
            sqlConstraint.ConstraintType = "PRIMARY KEY";
            _text.Match("KEY");
        }
        else if (constraintType.ToUpper() == "FOREIGN")
        {
            sqlConstraint.ConstraintType = "FOREIGN KEY";
            _text.Match("KEY");
        }
        else
        {
            return new Either<SqlConstraint?, ParseError>(
                new ParseError($"Expected PRIMARY KEY or FOREIGN KEY, but got {constraintType}"));
        }

        if (_text.TryMatch("CLUSTERED"))
        {
            sqlConstraint.Clustered = "CLUSTERED";
        }

        _text.Match("(");
        var indexColumns = new List<SqlColumnConstraint>();
        do
        {
            var indexColumn = new SqlColumnConstraint();
            indexColumn.ColumnName = _text.ReadSqlIdentifier().Word;
            if (_text.TryMatch("ASC"))
            {
                indexColumn.Order = "ASC";
            }
            else if (_text.TryMatch("DESC"))
            {
                indexColumn.Order = "DESC";
            }

            indexColumns.Add(indexColumn);
            if (_text.PeekChar() != ',')
            {
                break;
            }

            _text.NextChar();
        } while (!_text.IsEnd());

        _text.Match(")");
        sqlConstraint.Columns = indexColumns;
        if (_text.TryMatch("WITH"))
        {
            _text.Match("(");
            var toggles = new List<SqlWithToggle>();
            do
            {
                var toggle = new SqlWithToggle();
                toggle.ToggleName = _text.ReadSqlIdentifier().Word;
                _text.Match("=");
                toggle.Value = _text.ReadSqlIdentifier().Word;
                toggles.Add(toggle);
                if (_text.PeekChar() != ',')
                {
                    break;
                }

                _text.NextChar();
            } while (!_text.IsEnd());

            _text.Match(")");
            sqlConstraint.WithToggles = toggles;
        }

        if (_text.TryMatch("ON"))
        {
            sqlConstraint.On = _text.ReadSqlIdentifier().Word;
        }

        return new Either<SqlConstraint?, ParseError>(sqlConstraint);
    }

    public Either<ISqlExpression, ParseError> ParseSelectStatement()
    {
        if (!_text.TryMatchKeyword("SELECT"))
        {
            return CreateStartParseError(
                $"Expected SELECT, but got {_text.PreviousWord().Word} {_text.PeekKeyword().Word}");
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
                throw new ParseError("Expected column name");
            }

            if (_text.PeekChar() != ',')
            {
                break;
            }

            _text.NextChar();
        } while (!_text.IsEnd());

        var selectStatement = new SelectStatement
        {
            Columns = columns
        };

        if (_text.TryMatchKeyword("FROM"))
        {
            var tableName = _text.ReadIdentifier().Word;
            selectStatement.From = new SelectFrom()
            {
                FromTableName = tableName
            };
        }

        if (_text.TryMatchKeyword("WHERE"))
        {
            var left = ParseValue();
            var operation = _text.ReadSymbol().Word;
            var right = ParseValue();
            selectStatement.Where = new SqlWhereExpression()
            {
                Left = left,
                Operation = operation,
                Right = right
            };
        }

        return new Either<ISqlExpression, ParseError>(selectStatement);
    }

    public bool Try(Func<Either<ISqlExpression, ParseError>> parseFunc, out ISqlExpression sqlExpr,
        out ParseError error)
    {
        ISqlExpression localSqlExpr = new SqlEmptyExpression();
        var localError = ParseError.Empty;
        var rc = parseFunc();
        var success = rc.Match(left =>
            {
                localSqlExpr = left;
                return true;
            },
            right =>
            {
                localError = right;
                return false;
            });
        sqlExpr = localSqlExpr;
        error = localError;
        return success;
    }

    private Either<ISqlExpression, ParseError> CreateParseError(string message)
    {
        return new Either<ISqlExpression, ParseError>(new ParseError(message)
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

    private ColumnDefinition ParseDataDeColumnDefinition(TextSpan item)
    {
        var column = new ColumnDefinition
        {
            ColumnName = item.Word,
            DataType = _text.ReadSqlIdentifier().Word
        };

        var dataLength1 = string.Empty;
        var dataLength2 = string.Empty;
        if (_text.TryMatch("("))
        {
            dataLength1 = _text.ReadNumber().Word;
            dataLength2 = string.Empty;
            if (_text.PeekChar() == ',')
            {
                _text.NextChar();
                dataLength2 = _text.ReadNumber().Word;
            }

            _text.Match(")");
        }

        if (!string.IsNullOrEmpty(dataLength1))
        {
            column.Size = int.Parse(dataLength1);
        }

        if (!string.IsNullOrEmpty(dataLength2))
        {
            column.Scale = int.Parse(dataLength2);
        }

        return column;
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

    private SqlIdentity ParseSqlIdentity()
    {
        if (!_text.TryMatch("IDENTITY"))
        {
            return SqlIdentity.Default;
        }

        var sqlIdentity = new SqlIdentity
        {
            Seed = 1,
            Increment = 1
        };
        if (_text.TryMatch("("))
        {
            sqlIdentity.Seed = int.Parse(_text.ReadNumber().Word);
            _text.Match(",");
            sqlIdentity.Increment = int.Parse(_text.ReadNumber().Word);
            _text.Match(")");
        }

        return sqlIdentity;
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

    private ISqlExpression ParseValue()
    {
        if (Try(ParseIntValue, out var number, out _))
        {
            return number;
        }

        if (Try(ParseTableName, out var tableName, out _))
        {
            return tableName;
        }

        throw new ParseError("Expected Int");
    }
}

public interface ISqlConstraint
{
}

public class SqlConstraintDefault : ISqlConstraint
{
    public string ConstraintName { get; set; } = string.Empty;
    public string Value { get; set; } = string.Empty;
}