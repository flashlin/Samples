using Microsoft.EntityFrameworkCore;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq.Expressions;
using System.Reflection;
using System.Text;
using T1.Standard.Expressions;

namespace PizzaWeb.Models.Libs
{
	public static class DbSetExtension
	{
		public static UpdateOrm<T> Set<T, TValue>(this DbSet<T> dbSet, Expression<Func<T, TValue>> memberLamda, TValue? value)
			where T : class
		{
			return new UpdateOrm<T>(dbSet).Set(memberLamda, value);
		}
	}

	public class UpdateOrm<T>
		where T : class
	{
		private DbSet<T> _dbSet;
		private readonly Dictionary<string, object?> _setFields = new Dictionary<string, object?>();
		private string _where;

		public UpdateOrm(DbSet<T> dbSet)
		{
			this._dbSet = dbSet;
		}

		public UpdateOrm<T> Set<TValue>(Expression<Func<T, TValue>> memberLambda, TValue? value)
		{
			var simpleProperty = memberLambda.GetSimplePropertyAccess()
				.First();
			var name = simpleProperty.Name;
			_setFields[name] = value;
			return this;
		}

		public UpdateOrm<T> Where(Expression<Func<T, bool>> filterAction)
		{
			_where = new WhereExpressionVisitor<T>(null).GetWhere(filterAction);
			return this;
		}

		public void Update()
		{
			var tableAttr = typeof(T).GetCustomAttribute<TableAttribute>();
			var tableName = tableAttr?.Name ?? typeof(T).Name;

			var sb = new StringBuilder();
			sb.Append("UPDATE " + tableName + " as x");
			sb.AppendLine();
			foreach (var item in _setFields.Select((field, idx) => new { idx, field }))
			{
				if (item.idx != 0)
				{
					sb.Append(",");
					sb.AppendLine();
				}
				sb.Append(item.field.Key);
				sb.Append(" = ");
				sb.Append(item.field.Value);
			}
			sb.AppendLine();
			sb.Append("WHERE " + _where);
			var code = sb.ToString();
		}
	}

	public class WhereExpressionVisitor<T> : ExpressionVisitor
	{
		private TextWriter? _writer;

		public WhereExpressionVisitor(TextWriter? writer)
		{
			_writer = writer;
			if (_writer == null)
			{
				_writer = new StringWriter();
			}
		}

		protected override Expression VisitParameter(ParameterExpression node)
		{
			_writer.Write(node.Name);
			return node;
		}

		protected override Expression VisitLambda<T>(Expression<T> node)
		{
			_writer.Write('(');
			_writer.Write(string.Join(',', node.Parameters.Select(param => param.Name)));
			_writer.Write(')');
			_writer.Write("=>");
			Visit(node.Body);
			return node;
		}

		protected override Expression VisitConditional(ConditionalExpression node)
		{
			Visit(node.Test);

			_writer.Write('?');

			Visit(node.IfTrue);

			_writer.Write(':');

			Visit(node.IfFalse);

			return node;
		}

		protected override Expression VisitBinary(BinaryExpression node)
		{
			Visit(node.Left);

			_writer.Write(GetOperator(node.NodeType));

			Visit(node.Right);

			return node;
		}

		protected override Expression VisitMember(MemberExpression node)
		{
			// Closures are represented as a constant object with fields representing each closed over value.
			// This gets and prints the value of that closure.
			if (node.Member is FieldInfo fieldInfo && node.Expression is ConstantExpression constExpr)
			{
				WriteConstantValue(fieldInfo.GetValue(constExpr.Value)!);
			}
			else
			{
				Visit(node.Expression);
				_writer.Write('.');
				_writer.Write(node.Member.Name);
			}
			return node;
		}

		protected override Expression VisitConstant(ConstantExpression node)
		{
			WriteConstantValue(node.Value);

			return node;
		}

		private void WriteConstantValue(object obj)
		{
			switch (obj)
			{
				case string str:
					_writer.Write("'");
					_writer.Write(str);
					_writer.Write("'");
					break;
				default:
					_writer.Write(obj);
					break;
			}
		}

		private static string GetOperator(ExpressionType type)
		{
			switch (type)
			{
				case ExpressionType.Equal:
					return "=";
				case ExpressionType.Not:
					return "!";
				case ExpressionType.NotEqual:
					return "<>";
				case ExpressionType.GreaterThan:
					return ">";
				case ExpressionType.GreaterThanOrEqual:
					return ">=";
				case ExpressionType.LessThan:
					return "<";
				case ExpressionType.LessThanOrEqual:
					return "<=";
				case ExpressionType.Or:
					return " or ";
				case ExpressionType.OrElse:
					return " OR";
				case ExpressionType.And:
					return " and ";
				case ExpressionType.AndAlso:
					return " AND ";
				case ExpressionType.Add:
					return "+";
				case ExpressionType.AddAssign:
					return "+=";
				case ExpressionType.Subtract:
					return "-";
				case ExpressionType.SubtractAssign:
					return "-=";
				default:
					return "???";
			}
		}


		public string GetWhere(Expression<Func<T, bool>> whereExpression)
		{
			this.Visit(whereExpression.Body);
			var s = _writer.ToString();
			return s!;
		}
	}
}
