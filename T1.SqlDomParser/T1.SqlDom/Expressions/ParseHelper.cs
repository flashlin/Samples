namespace T1.SqlDom.Expressions;

public delegate ParseResult ParseFunc(InputStream inp);

public static class ParseHelper
{
    public static SqlExpr Parse(this ParseFunc parseFunc, InputStream inp, string expect)
    {
        var position = inp.Position;
        var rc = parseFunc(inp);
        if (!rc.Success)
        {
            inp.Position = position;
            var subStr = inp.SubStr(inp.Position, 20);
            throw new Exception($"Expect '{expect}' at Position {inp.Position}, '{subStr}'");
        }
        return rc.Expr;
    }
    
    public static ParseResult Any(ParseFunc[] parseFuncList, InputStream inp)
    {
        foreach (var parseFunc in parseFuncList)
        {
            if (parseFunc.Try(inp, out var rc))
            {
                return rc;
            }
        }
        return ParseResult.Empty;
    }
    
    
    public static MatchParseFuncResult MatchAny(InputStream inp, params ParseFunc[] parseFuncList)
    {
        foreach (var item in parseFuncList.Select((parseFunc,idx) => (parseFunc, idx)))
        {
            if (item.parseFunc.Try(inp, out var rc))
            {
                return new MatchParseFuncResult
                {
                    Func = item.parseFunc,
                    Expr = rc.Expr,
                };
            }
        }
        return new MatchParseFuncResult
        {
            Func = null,
            Expr = SqlExpr.Empty,
        };
    }
    
    
    public static bool TryBindAny(ParseFunc[] parseFuncList, InputStream inp, Action<SqlExpr> bind)
    {
        var match = MatchAny(inp, parseFuncList);
        if (match.Func == null)
        {
            return false;
        }
        bind(match.Expr);
        return true;
    }


    public static ParseResult Match(ParseFunc parseFunc, InputStream inp, Func<ParseResult, ParseResult> mapFunc)
    {
        var position = inp.Position;
        var rc = parseFunc(inp);
        if (!rc.Success)
        {
            inp.Position = position;
            return rc;
        }
        rc = mapFunc(rc);
        if (!rc.Success)
        {
            inp.Position = position;
        }
        return rc;
    }

    public static bool Try(this ParseFunc parseFunc, InputStream inp, out ParseResult result)
    {
        var position = inp.Position;
        result = parseFunc(inp);
        if (!result.Success)
        {
            inp.Position = position;
            return false;
        }
        return true;
    }
}