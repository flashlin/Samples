using System.Collections.Immutable;
using System.Runtime.InteropServices.ComTypes;
using Pidgin;
using Pidgin.Expression;
using static Pidgin.Parser;

//https://github.com/benjamin-hodgson/Pidgin/blob/main/Pidgin.Examples/Expression/ExprParser.cs


namespace PidginDemo.LinqExpressions;

public static class LinqParser
{
    private static Parser<char, T> Tok<T>(Parser<char, T> token)
        => Try(token).Before(SkipWhitespaces);

    private static Parser<char, string> Tok(string token)
        => Tok(String(token));
    
    private static Parser<char, string> IdentifierPrefix
        => OneOf(
            Letter,
            Char('_')
        ).ManyString();

    private static Parser<char, string> IdentifierPostfix
        => OneOf(
            LetterOrDigit,
            Char('_')
        ).ManyString();
        
    
    private static readonly Parser<char, LinqExpr> Identifier
        = Tok(IdentifierPrefix.Then(IdentifierPostfix, (h, t) => h + t))
            .Select<LinqExpr>(name => new IdentifierExpr(name))
            .Labelled("identifier");
    
    private static readonly Parser<char, LinqExpr> Integer
        = Tok(Num)
            .Select<LinqExpr>(value => new IntegerExpr(value))
            .Labelled("integer literal");
    
    private static Parser<char, T> Parenthesised<T>(Parser<char, T> parser)
        => parser.Between(Tok("("), Tok(")"));
    
    private static Parser<char, Func<LinqExpr, LinqExpr>> Call(Parser<char, LinqExpr> subExpr)
        => Parenthesised(subExpr.Separated(Tok(",")))
            .Select<Func<LinqExpr, LinqExpr>>(args => method => new CallExpr(method, args.ToImmutableArray()))
            .Labelled("function call");
    
    private static Parser<char, Func<LinqExpr, LinqExpr, LinqExpr>> Binary(Parser<char, BinaryOperatorType> op)
        => op.Select<Func<LinqExpr, LinqExpr, LinqExpr>>(type => (l, r) => new BinaryOpExpr(type, l, r));
    
    private static Parser<char, Func<LinqExpr, LinqExpr>> Unary(Parser<char, UnaryOperatorType> op)
        => op.Select<Func<LinqExpr, LinqExpr>>(type => o => new UnaryOpExpr(type, o));
    
    private static readonly Parser<char, Func<LinqExpr, LinqExpr, LinqExpr>> AddOp
        = Binary(Tok("+").ThenReturn(BinaryOperatorType.Add));
    
    private static readonly Parser<char, Func<LinqExpr, LinqExpr, LinqExpr>> MulOp
        = Binary(Tok("*").ThenReturn(BinaryOperatorType.Mul));
    
    private static readonly Parser<char, Func<LinqExpr, LinqExpr>> ComplementOp
        = Unary(Tok("~").ThenReturn(UnaryOperatorType.Complement));
    
    private static readonly Parser<char, Func<LinqExpr, LinqExpr>> NegOp
        = Unary(Tok("-").ThenReturn(UnaryOperatorType.Neg));

    private static readonly Parser<char, string> FROM
        = Tok("from");

    private static readonly Parser<char, string> IN
        = Tok("in");
    
    private static readonly Parser<char, LinqExpr> SelectExpr
        = Map(
            (_, identifier, _) => new SelectExpr {
                AliasTable = (identifier as IdentifierExpr)!.Name,
            } as LinqExpr, 
            FROM,
            Identifier,
            IN
        );

    private static readonly Parser<char, LinqExpr> Expr = ExpressionParser.Build<char, LinqExpr>(
        expr => (
            OneOf(
                Identifier,
                Integer,
                Parenthesised(expr).Labelled("parenthesised expression")
            ),
            new[]
            {
                Operator.PostfixChainable(Call(expr)),
                Operator.Prefix(NegOp).And(Operator.Prefix(ComplementOp)),
                Operator.InfixL(MulOp),
                Operator.InfixL(AddOp)
            }
        )
    ).Labelled("expression");
    
    public static LinqExpr ParseOrThrow(string input)
        => SelectExpr.ParseOrThrow(input);
}