using T1.ParserKit.Core.Parsers;
using T1.ParserKit.Core.TextUtils;

namespace T1.ParserKit.Core;

public class LinqGrammar : Grammar<LinqSyntaxNode>
{
    private Parser<Text, LinqSelectExpr>? _linqSelectExpr;
    public virtual Parser<Text, LinqSelectExpr> SelectExpr
    {
        get
        {
            if (_linqSelectExpr == null)
            {
                _linqSelectExpr =
                    from fromKeyword in String("from")
                    from space in WhiteSpace
                    from aliasTableName in Identifier
                    select new LinqSelectExpr()
                    {
                        AliasTableName = aliasTableName
                    };
            }
            return _linqSelectExpr;
        }
    }

    private Parser<Text, Text>? _identifier;
    public Parser<Text, Text> Identifier
    {
        get
        {
            if (_identifier == null)
            {
                _identifier = from prefix in Letter | Char('_')
                    from postfix in (Letter | Digit).ZeroOrMore()
                    select prefix.Append(postfix);
            }
            return _identifier;
        }
    }

    private Parser<Text, LinqSyntaxNode>? _parser;
    public override Parser<Text, LinqSyntaxNode> Parser
    {
        get
        {
            if (_parser == null)
            {
                _parser = from p in SelectExpr 
                    select p as LinqSyntaxNode;
            }
            return _parser;
        }
    }

    public LinqSyntaxNode? Parse(string text)
    {
        //var grammar = new LinqGrammar();
        var node = Parser.Parse(text);
        return node.Value;
    }
}

public abstract class LinqSyntaxNode
{
}

public class LinqSelectExpr : LinqSyntaxNode
{
    public string AliasTableName { get; set; }
}
