namespace T1.SqlSharp.Expressions;

public interface ITableSource : ISqlExpression
{
    string Alias { get; set; }
    List<ISqlExpression> Withs { get; set; }
    List<SqlJoinTableCondition> JoinTables { get; set; }
}