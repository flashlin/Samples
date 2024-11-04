using System.Reflection;
using Sprache;

namespace SqlSharpLit.SqlExpressions;

public interface ISqlParserProvider
{
    IComment Parser { get; }
}

public static class SqlKeywords
{
    public const string Create = "create";
    public const string Table = "table";
    public const string Not = "not";
    public const string Null = "null";
    public const string Identity = "identity";
    public const string Int = "int";
    public const string Decimal = "decimal";
    public const string Nvarchar = "nvarchar";
    public const string Varchar = "varchar";
    public const string Datetime = "datetime";
    public const string Constraint = "constraint";
    public const string Default = "default";
    public const string GetDate = "getdate";
    
    private static IEnumerable<string> AllStringConstants =>
        typeof(SqlKeywords).GetTypeInfo().GetFields().Select(f => f.GetValue(null)).OfType<string>();
    
    public static HashSet<string> ReservedWords { get; } =
        GetStrings(AllStringConstants.ToArray());

    private static HashSet<string> GetStrings(params string[] strings)
    {
        return new HashSet<string>(strings, StringComparer.OrdinalIgnoreCase);
    }
}

public class SqlCreateTableGrammar : ISqlParserProvider
{
    public IComment Parser { get; } = new CommentParser();
    
    protected virtual Parser<string> RawIdentifier =>
        from identifier in Parse.Identifier(Parse.Letter, Parse.LetterOrDigit.Or(Parse.Char('_')))
        where !SqlKeywords.ReservedWords.Contains(identifier)
        select identifier;

    protected virtual Parser<string> Identifier =>
        RawIdentifier.Token().Named("Identifier");

    protected internal virtual Parser<IEnumerable<string>> QualifiedIdentifier =>
        Identifier.DelimitedBy(Parse.Char('.').Token())
            .Named("QualifiedIdentifier");
    
    protected Parser<IEnumerable<string>> ColumnName =>
        Identifier.DelimitedBy(Parse.Char(',').Token())
            .Named("ColumnName");
    
    public Parser<SqlCreateTable> CreateTableStatement =>
        from create in Parse.IgnoreCase(SqlKeywords.Create).Token()
        from table in Parse.IgnoreCase(SqlKeywords.Table).Token()
        from tableName in Identifier
        from columns in ColumnName.Many()
        select new SqlCreateTable
        {
            
        };
}

public class SqlCreateTable
{
}

public static class SqlParser
{
    public static SqlCreateTable ParseCreateTable(string text)
    {
        return new SqlCreateTableGrammar().CreateTableStatement.Parse(text);
    }
}