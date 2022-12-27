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
            var rc = parseFunc(inp);
            if (rc.Success)
            {
                return rc;
            }
        }
        return ParseResult.Empty;
    }
}