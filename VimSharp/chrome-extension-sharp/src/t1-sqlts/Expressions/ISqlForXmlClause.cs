namespace T1.SqlSharp.Expressions;

public interface ISqlForXmlClause : ISqlExpression
{
    List<SqlForXmlRootDirective> CommonDirectives { get; set; }
}


