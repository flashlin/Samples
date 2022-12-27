using T1.SqlDomParser;

namespace T1.SqlDom.Expressions;

public class SqlParser
{
    public SqlExpr Parse(string input)
    {
        return ParseExpr(new InputStream
        {
            Value = input,
            Position = 0
        });
    }

    private SqlExpr ParseExpr(InputStream inputStream)
    {
        return ParseSelectStatement(inputStream);
    }

    public SelectExpr ParseSelectStatement(InputStream inp)
    {
        var statement = new SelectExpr();
        inp.ExpectKeyword("SELECT");

        statement.Columns = ParseColumns(inp);

        inp.ExpectKeyword("FROM");

        statement.Tables = ParseTables(inp);

        if (inp.AcceptKeyword("WHERE"))
        {
            statement.WhereClause = ParseWhereClause(inp);
        }

        return statement;
    }

    private SqlExpr ParseWhereClause(InputStream inp)
    {
        return ParseOperatorExpr(inp);
    }

    private ComparsionExpr ParseOperatorExpr(InputStream inp)
    {
        var left = ParseHelper.Parse(ParseConstant, inp, "Left Expr");
        var op = ParseHelper.Parse(ParseOperator, inp, "operator");
        var right = ParseHelper.Parse(ParseConstant, inp, "Right Expr");
        return new ComparsionExpr
        {
            Left = left,
            Oper = op,
            Right = right,
        };
    }

    private ParseResult ParseConstant(InputStream inp)
    {
        return ParseHelper.Any(new ParseFunc[]
        {
            ParseColumn,
            ParseNumber
        }, inp);
    }

    private ParseResult ParseNumber(InputStream inp)
    {
        inp.SkipSpaces();
        if (!inp.Accept(c => char.IsNumber(c), out var ch))
        {
            return ParseResult.Empty;
        }

        var number = string.Empty;
        number += ch;
        var hasDot = false;
        while (inp.Accept(c => char.IsNumber(c) || (c == '.' && !hasDot), out ch))
        {
            if (ch == '.')
            {
                hasDot = true;
            }

            number += ch;
        }

        return new ParseResult
        {
            Expr = new NumberExpr
            {
                Value = number
            },
            Success = true,
        };
    }

    private ParseResult ParseOperator(InputStream inp)
    {
        if (!inp.AcceptAnyKeyword(new[] { "==", "<=", ">=", "<", ">", "!=", "+", "-", "*", "/", "%" }, out var op))
        {
            return ParseResult.Empty;
        }

        return new ParseResult
        {
            Expr = new StringExpr
            {
                Value = op,
            },
            Success = true
        };
    }


    private List<TableExpr> ParseTables(InputStream inp)
    {
        var tables = new List<TableExpr>();
        tables.Add(ParseTable(inp));
        while (inp.Accept(','))
        {
            tables.Add(ParseTable(inp));
        }

        return tables;
    }

    private TableExpr ParseTable(InputStream inp)
    {
        if (inp.Accept('('))
        {
            var subQuery = ParseExpr(inp);
            inp.Expect(')');
            var subTable = new TableExpr
            {
                Name = subQuery,
                IsSubQuery = true
            };
            if (inp.AcceptKeyword("AS"))
            {
                subTable.Alias = ParseHelper.Parse(ParseIdentifier, inp, "AliasName");
            }

            return subTable;
        }

        var table = new TableExpr
        {
            Name = ParseHelper.Parse(ParseIdentifier, inp, "FieldName"),
            IsSubQuery = true
        };
        if (inp.AcceptKeyword("AS"))
        {
            table.Alias = ParseHelper.Parse(ParseIdentifier, inp, "AliasName");
        }

        return table;
    }

    private List<SqlExpr> ParseColumns(InputStream inp)
    {
        var columns = new List<SqlExpr>();
        columns.Add(ParseHelper.Parse(ParseColumn, inp, "Column"));
        while (inp.Accept(','))
        {
            columns.Add(ParseHelper.Parse(ParseColumn, inp, "Column"));
        }

        return columns;
    }

    private ParseResult ParseColumn(InputStream inp)
    {
        inp.SkipSpaces();
        var column = new ColumnExpr
        {
            Name = ParseHelper.Parse(ParseIdentifier, inp, "Column"),
        };

        if (inp.AcceptKeyword("AS"))
        {
            column.Alias = ParseHelper.Parse(ParseIdentifier, inp, "AliasName");
        }

        return new ParseResult()
        {
            Expr = column,
            Success = true,
        };
    }


    private ParseResult ParseIdentifier(InputStream inp)
    {
        inp.SkipSpaces();
        var identifier = "";
        if (!inp.Accept(c => char.IsLetter(c) || c == '_', out var ch))
        {
            return ParseResult.Empty;
        }

        identifier += ch;
        while (inp.Accept(c => char.IsLetter(c) || c == '_', out ch))
        {
            identifier += ch;
        }

        return new ParseResult
        {
            Expr = new StringExpr
            {
                Value = identifier
            },
            Success = true,
        };
    }
}