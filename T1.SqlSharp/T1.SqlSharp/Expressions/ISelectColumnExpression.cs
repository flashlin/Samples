namespace T1.SqlSharp.Expressions;

public interface ISelectColumnExpression : ISqlExpression 
{
    string Alias { get; set; }
}