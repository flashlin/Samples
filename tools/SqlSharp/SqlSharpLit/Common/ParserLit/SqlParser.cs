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
        if (Try(ParseCreateTableStatement, out var sqlCreateTableExpr))
        {
            return new Either<ISqlExpression, ParseError>(sqlCreateTableExpr);
        }

        if (Try(ParseSelectStatement, out var sqlSelectExpr))
        {
            return new Either<ISqlExpression, ParseError>(sqlSelectExpr);
        }

        return new Either<ISqlExpression, ParseError>(new ParseError("Unknown statement"));
    }

    public Either<ISqlExpression, ParseError> ParseCreateTableStatement()
    {
        if (!(_text.TryMatchKeyword("CREATE") && _text.TryMatchKeyword("TABLE")))
        {
            return new Either<ISqlExpression, ParseError>(
                new ParseError(
                    $"Expected CREATE TABLE, but got {_text.PreviousWord().Word} {_text.PeekKeyword().Word}"));
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

            var column = new ColumnDefinition()
            {
                ColumnName = item.Word,
            };

            column.DataType = _text.ReadSqlIdentifier().Word;
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

    public Either<ISqlExpression, ParseError> ParseSelectStatement()
    {
        if (!_text.TryMatchKeyword("SELECT"))
        {
            return new Either<ISqlExpression, ParseError>(
                new ParseError($"Expected SELECT, but got {_text.PreviousWord().Word} {_text.PeekKeyword().Word}"));
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
            var right= ParseValue();
            selectStatement.Where = new SqlWhereExpression()
            {
                Left = left,
                Operation = operation,
                Right = right
            };   
        }
        return new Either<ISqlExpression, ParseError>(selectStatement);
    }

    public bool Try(Func<Either<ISqlExpression, ParseError>> parseFunc, out ISqlExpression sqlExpr)
    {
        return ParseHelper.Try(parseFunc, out sqlExpr);
    }

    private Either<ISqlExpression,ParseError> ParseIntValue()
    {
        if (_text.Try(_text.ReadNumber, out var number))
        {
            return new Either<ISqlExpression, ParseError>(new SqlIntValueExpression
            {
                Value = int.Parse(number.Word)
            });
        }
        return new Either<ISqlExpression,ParseError>(new ParseError("Expected Int"));
    }

    private Either<ISqlExpression,ParseError> ParseTableName()
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
        if (Try(ParseIntValue, out var number))
        {
            return number;
        }
        if(Try(ParseTableName, out var tableName))
        {
            return tableName;
        }
        throw new ParseError("Expected Int");
    }
}

public static class ParseHelper
{
    public static bool Try(Func<Either<ISqlExpression, ParseError>> parseFunc, out ISqlExpression sqlExpr)
    {
        ISqlExpression localSqlExpr = new SqlEmptyExpression();
        var rc = parseFunc();
        var success = rc.Match(left =>
            {
                localSqlExpr = left;
                return true;
            },
            right => false);
        sqlExpr = localSqlExpr;
        return success;
    }
}

public class SqlEmptyExpression : ISqlExpression
{}