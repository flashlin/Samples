namespace T1.SqlSharp.Expressions;

public interface ISqlConstraint : ISqlExpression
{
    string ConstraintName { get; set; }
}