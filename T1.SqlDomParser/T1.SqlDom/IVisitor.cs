namespace T1.SqlDomParser
{
	public interface IVisitor
	{
		void Visit(BinaryExpr unaryExpr);
		void Visit(NumberLiteral numberLiteral);
	}
}
