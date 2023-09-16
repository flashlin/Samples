using System.Collections.Immutable;
using Pidgin;
using Pidgin.Expression;
using static Pidgin.Parser;

//https://github.com/benjamin-hodgson/Pidgin/blob/main/Pidgin.Examples/Expression/ExprParser.cs

public enum BinaryOperatorType
{
    Add,
    Sub,
    Mul,
    Divide
}


public static class TSqlParser
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
        
    
    private static readonly Parser<char, SqlExpr> Identifier
        = Tok(IdentifierPrefix.Then(IdentifierPostfix, (h, t) => h + t))
            .Select<SqlExpr>(name => new IdentifierExpr(name))
            .Labelled("identifier");
    
    private static readonly Parser<char, SqlExpr> Integer
        = Tok(Num)
            .Select<SqlExpr>(value => new IntegerExpr(value))
            .Labelled("integer literal");
    
    private static Parser<char, T> Parenthesised<T>(Parser<char, T> parser)
        => parser.Between(Tok("("), Tok(")"));
    
    private static Parser<char, Func<SqlExpr, SqlExpr>> Call(Parser<char, SqlExpr> subExpr)
        => Parenthesised(subExpr.Separated(Tok(",")))
            .Select<Func<SqlExpr, SqlExpr>>(args => method => new CallExpr(method, args.ToImmutableArray()))
            .Labelled("function call");
    
    private static Parser<char, Func<SqlExpr, SqlExpr, SqlExpr>> Binary(Parser<char, BinaryOperatorType> op)
        => op.Select<Func<SqlExpr, SqlExpr, SqlExpr>>(type => (l, r) => new BinaryOpExpr(type, l, r));
    
    private static Parser<char, Func<SqlExpr, SqlExpr>> Unary(Parser<char, UnaryOperatorType> op)
        => op.Select<Func<SqlExpr, SqlExpr>>(type => o => new UnaryOpExpr(type, o));
    
    private static readonly Parser<char, Func<SqlExpr, SqlExpr, SqlExpr>> AddOp
        = Binary(Tok("+").ThenReturn(BinaryOperatorType.Add));
    
    private static readonly Parser<char, Func<SqlExpr, SqlExpr, SqlExpr>> MulOp
        = Binary(Tok("*").ThenReturn(BinaryOperatorType.Mul));
    
    private static readonly Parser<char, Func<SqlExpr, SqlExpr>> ComplementOp
        = Unary(Tok("~").ThenReturn(UnaryOperatorType.Complement));
    
    private static readonly Parser<char, Func<SqlExpr, SqlExpr>> NegOp
        = Unary(Tok("-").ThenReturn(UnaryOperatorType.Neg));

    private static readonly Parser<char, SqlExpr> Expr = ExpressionParser.Build<char, SqlExpr>(
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
    
    public static SqlExpr ParseOrThrow(string input)
        => Expr.ParseOrThrow(input);
}