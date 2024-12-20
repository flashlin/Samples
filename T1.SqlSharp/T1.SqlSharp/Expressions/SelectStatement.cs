using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SelectStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.SelectStatement;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_SelectStatement(this);
    }

    public SelectType SelectType { get; set; } = SelectType.None; 
    public SqlTopClause? Top { get; set; }
    public List<ISelectColumnExpression> Columns { get; set; } = [];
    public List<ISqlExpression> FromSources { get; set; } = [];
    public ISqlForXmlClause? ForXml { get; set; }
    public ISqlExpression? Where { get; set; }
    public SqlOrderByClause? OrderBy { get; set; }
    public List<SqlUnionSelect> Unions { get; set; } = [];
    public SqlGroupByClause? GroupBy { get; set; }
    public SqlHavingClause? Having { get; set; }

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write("SELECT");
        if(SelectType!=SelectType.None)
        {
            sql.Write($" {SelectType.ToString().ToUpper()} ");
        }
        sql.WriteLine();
        if(Top!=null)
        {
            sql.Write(Top.ToSql());
            sql.WriteLine();
        }
        sql.Indent++;
        for (var i = 0; i < Columns.Count; i++)
        {
            sql.Write(Columns[i].ToSql());
            if (i < Columns.Count - 1)
            {
                sql.Write(",");
            }
            sql.WriteLine();
        }
        sql.Indent--;
        if (FromSources.Count > 0)
        {
            sql.WriteLine("FROM ");
            sql.Indent++;
            foreach (var item in FromSources.Select((tableSource, index) => new { tableSource, index }))
            {
                sql.Write(item.tableSource.ToSql());
                if (item.index < FromSources.Count - 1)
                {
                    sql.WriteLine(",");
                }
            }
            sql.Indent--;
            
            if(ForXml!=null)
            {
                sql.WriteLine(ForXml.ToSql());
            }
        }
        sql.Indent++;
        if(Where!=null)
        {
            sql.WriteLine("WHERE ");
            sql.Write(Where.ToSql());
        }
        sql.Indent--;
        if(GroupBy!=null)
        {
            sql.WriteLine(" " + GroupBy.ToSql());
        }
        if(OrderBy!=null)
        {
            sql.WriteLine(" " + OrderBy.ToSql());
        }
        foreach (var unionSelect in Unions)
        {
            sql.WriteLine(unionSelect.ToSql());
        }
        if(Having!=null)
        {
            sql.WriteLine("HAVING ");
            sql.Write(Having.ToSql());
        }
        return sql.ToString();
    }
}