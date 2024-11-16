using T1.Standard.DesignPatterns;

namespace SqlSharpLit.Common.ParserLit;

public class SqlParser
{
    private readonly StringParser _text;

    public SqlParser(string text)
    {
        _text = new StringParser(text);
    }

    public Either<ISqlExpression, ParseError> Parse()
    {
        var error = ParseError.Empty;
        if (Try(ParseCreateTableStatement, out var sqlCreateTableExpr, out error))
        {
            return new Either<ISqlExpression, ParseError>(sqlCreateTableExpr);
        }

        if (!error.IsStart)
        {
            return new Either<ISqlExpression, ParseError>(error);
        }

        if (Try(ParseSelectStatement, out var sqlSelectExpr,out error))
        {
            return new Either<ISqlExpression, ParseError>(sqlSelectExpr);
        }
        
        if (!error.IsStart)
        {
            return new Either<ISqlExpression, ParseError>(error);
        }

        return new Either<ISqlExpression, ParseError>(new ParseError("Unknown statement"));
    }

    public Either<ISqlExpression, ParseError> ParseCreateTableStatement()
    {
        if (!(_text.TryMatchKeyword("CREATE") && _text.TryMatchKeyword("TABLE")))
        {
            return new Either<ISqlExpression, ParseError>(
                new ParseError(
                    $"Expected CREATE TABLE, but got {_text.PreviousWord().Word} {_text.PeekKeyword().Word}")
                {
                    IsStart = true
                });
        }

        var tableName = _text.ReadUntil(c => char.IsWhiteSpace(c) || c == '(');
        _text.Match("(");

        var columns = new List<ColumnDefinition>();
        do
        {
            var item = _text.ReadSqlIdentifier();
            if (item.Length == 0)
            {
                return new Either<ISqlExpression, ParseError>(
                    new ParseError($"Expected column name, but got {_text.PeekKeyword().Word}"));
            }

            var column = ParseDataDeColumnDefinition(item);
            column.Identity = ParseSqlIdentity();
            column.IsNullable = ParseDeclareNullable();
            columns.Add(column);
            if (_text.PeekChar() != ',')
            {
                break;
            }

            _text.NextChar();
        } while (!_text.IsEnd());

        return new Either<ISqlExpression, ParseError>(new CreateTableStatement()
        {
            TableName = tableName.Word,
            Columns = columns
        });
    }

    private bool ParseDeclareNullable()
    {
        if(_text.TryMatch("NOT"))
        {
            _text.Match("NULL");
            return false;
        }
        if(_text.TryMatch("NULL"))
        {
            return true;
        }
        return false;
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

    public Either<ISqlExpression, ParseError> ParseSelectStatement()
    {
        if (!_text.TryMatchKeyword("SELECT"))
        {
            return new Either<ISqlExpression, ParseError>(
                new ParseError($"Expected SELECT, but got {_text.PreviousWord().Word} {_text.PeekKeyword().Word}")
                {
                    IsStart = true
                });
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

    public bool Try(Func<Either<ISqlExpression, ParseError>> parseFunc, out ISqlExpression sqlExpr, out ParseError error)
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
        if (Try(ParseIntValue, out var number,out _))
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