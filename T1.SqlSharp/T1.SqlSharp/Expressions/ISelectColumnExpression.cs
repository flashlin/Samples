namespace T1.SqlSharp.Expressions;

public interface ISelectColumnExpression : ISqlExpression 
{
    ISqlExpression Field { get; set; }
    string Alias { get; set; }
}