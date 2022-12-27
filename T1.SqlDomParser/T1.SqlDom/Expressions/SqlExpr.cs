namespace T1.SqlDom.Expressions
{
	public abstract class SqlExpr
	{
		public static SqlExpr Empty = new EmptySqlExpr();
		public abstract string ToSqlString();
	}

	public class EmptySqlExpr : SqlExpr
	{
		public override string ToSqlString()
		{
			return string.Empty;
		}
	}
}
