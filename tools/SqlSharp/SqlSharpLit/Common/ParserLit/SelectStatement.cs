namespace SqlSharpLit.Common.ParserLit;

public class SelectStatement : ISqlExpression
{
    public List<ISelectColumnExpression> Columns { get; set; } = [];
    public ISelectFromExpression From { get; set; } = new SelectFrom();
    public ISqlWhereExpression Where { get; set; }
}