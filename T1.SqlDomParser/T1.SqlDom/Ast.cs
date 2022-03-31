using Superpower.Model;
using System;
using System.Collections.Generic;
using System.Text;

namespace T1.SqlDomParser
{
	public interface ISqlDom
	{
		void Accept(IVisitor visitor);
		TextSpan? Span { get; }
	}

	public enum ValueType
	{
		None,
		Scalar,
		String,
	}

	public interface SqlExpr : ISqlDom
	{
		//ValueType Type { get; }
		public string ToSqlCode();
	}

	public record BinaryExpr(SqlExpr leftSide, SqlExpr rightSide, Operators.Binary oper, TextSpan? Span = null) : SqlExpr
   {
		public SqlExpr LeftSide { get; set; } = leftSide;
		public SqlExpr RightSide { get; set; } = rightSide;
		public Operators.Binary Operator { get; set; } = oper;
		public void Accept(IVisitor visitor) => visitor.Visit(this);
		//public ValueType Type => ValueType.Scalar;

		public string ToSqlCode()
		{
			var sb = new StringBuilder();
			sb.Append($"{LeftSide.ToSqlCode()}");

			switch(Operator)
			{
				case Operators.Binary.Add:
					sb.Append(" +");
					break;
				case Operators.Binary.Sub:
					sb.Append(" -");
					break;
				case Operators.Binary.Mul:
					sb.Append(" *");
					break;
			}

			sb.Append($" {RightSide.ToSqlCode()}");
			return sb.ToString();
		}
	}

	public record NumberLiteral(decimal Value, TextSpan? Span = null) : SqlExpr
	{
		public void Accept(IVisitor visitor) => visitor.Visit(this);

		public string ToSqlCode()
		{
			var sb = new StringBuilder();
			sb.Append($"{Value}");
			return sb.ToString();
		}
	}
}
