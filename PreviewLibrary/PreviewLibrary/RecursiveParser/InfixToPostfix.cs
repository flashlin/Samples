using System;
using System.Collections.Generic;
using System.Linq;

namespace PreviewLibrary
{
	public class InfixToPostfix<T>
	{
		private Dictionary<string, int> _operatorsPriority = new Dictionary<string, int>();
		private readonly InfixToPostfixOptions<T> _options;

		public InfixToPostfix(InfixToPostfixOptions<T> options)
		{
			this._options = options;
			foreach (var item in _options.Operators.Reverse().Select((value, index) => new { value, index }))
			{
				_operatorsPriority[item.value] = item.index + 1;
			}
		}

		public bool IsOperator(string token)
		{
			return _operatorsPriority.ContainsKey(token);
		}
		
		public T Parse()
		{
			var postfixExprs = new List<object>();
			var ops = new Stack<string>();
			do
			{
				postfixExprs.Add(_options.ReadToken());
				var operatorToken = _options.GetCurrentTokenText();
				if (!IsOperator(operatorToken))
				{
					break;
				}
				_options.NextToken();
				if (ops.Count >= 1)
				{
					var prev_op = ops.Pop();
					if (CompareOp(prev_op, operatorToken) > 0)
					{
						ops.Push(prev_op);
						postfixExprs.Add(operatorToken);
					}
					else
					{
						ops.Push(operatorToken);
						postfixExprs.Add(prev_op);
					}
				}
				else
				{
					ops.Push(operatorToken);
				}
			} while (true);

			if (ops.Count > 0)
			{
				postfixExprs.Add(ops.Pop());
			}

			return ComputePostfixExprs(postfixExprs);
		}

		private T ComputePostfixExprs(List<object> postfixExprs)
		{
			var st = new Stack<T>();
			var iter = postfixExprs.GetEnumerator();
			while (iter.MoveNext())
			{
				var item = iter.Current;
				if (item is T sqlExpr)
				{
					st.Push(sqlExpr);
				}
				else
				{
					var oper = item as string;
					var right = st.Pop();
					var left = st.Pop();
					var expr = _options.CreateBinaryOperator(left, oper, right);
					st.Push(expr);
				}
			}

			if (st.Count > 1)
			{
				throw new Exception();
			}
			return st.Pop();
		}

		private int CompareOp(string op1, string op2)
		{
			var op1Priotity = _operatorsPriority[op1];
			var op2Priotity = _operatorsPriority[op2];
			if (op1Priotity == op2Priotity)
			{
				return 0;
			}
			if (op1Priotity > op2Priotity)
			{
				return 1;
			}
			return -1;
		}
	}
}
