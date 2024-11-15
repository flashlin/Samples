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
        if(Try(ParseCreateTableStatement, out var sqlExpr))
        {
            return new Either<ISqlExpression, ParseError>(sqlExpr);
        }
        return new Either<ISqlExpression, ParseError>(new ParseError("Unknown statement"));
    }

    public bool Try(Func<Either<CreateTableStatement, ParseError>> parseFunc, out ISqlExpression sqlExpr)
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

    public Either<CreateTableStatement, ParseError> ParseCreateTableStatement()
    {
        if (!(_text.TryMatchKeyword("CREATE") && _text.TryMatchKeyword("TABLE")))
        {
            return new Either<CreateTableStatement, ParseError>(
                new ParseError($"Expected CREATE TABLE, but got {_text.PreviousWord().Word} {_text.PeekKeyword().Word}"));
        }

        var tableName = _text.ReadUntil(c => char.IsWhiteSpace(c) || c == '(');
        _text.Match("(");

        var columns = new List<ColumnDefinition>();
        do
        {
            var item = _text.ReadIdentifier();
            if (item.Length == 0)
            {
                return new Either<CreateTableStatement, ParseError>(
                    new ParseError($"Expected column name, but got {_text.PeekKeyword().Word}"));
            }

            var column = new ColumnDefinition()
            {
                ColumnName = item.Word,
            };
            
            column.DataType = _text.ReadIdentifier().Word;
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

        return new Either<CreateTableStatement, ParseError>(new CreateTableStatement()
        {
            TableName = tableName.Word,
            Columns = columns
        });
    }
}

public class SqlEmptyExpression : ISqlExpression
{
}