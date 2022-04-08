using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
{
	public class LikeExpr : SqlExpr
	{
		public SqlExpr Left { get; set; }
		public string Right { get; set; }

		public override string ToString()
		{
			return $"{Left} LIKE {Right}";
		}
	}

	public class SetBatchVariableExpr : SqlExpr
	{
		public string Name { get; set; }
		public string Value { get; set; }
		public override string ToString()
		{
			return $":SETVAR {Name} {Value}";
		}
	}

	public class OnConditionThenExpr : SqlExpr
	{
		public string Condition { get; set; }
		public string ActionName { get; set; }

		public override string ToString()
		{
			return $":ON {Condition} {ActionName}";
		}
	}
}