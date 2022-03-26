using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Text;
using System.Text.RegularExpressions;
using T1.Standard.Extensions;

namespace PreviewLibrary
{
	public class SqlParser
	{
		private SqlTokenizer _token;
		private string _sql;
		private InfixToPostfix<SqlExpr> _arithmetic;

		public SqlParser()
		{
			Initialize();
			_token = new SqlTokenizer();
		}

		public SqlExpr Parse(string sql)
		{
			_sql = sql;
			_token.PredicateParse(sql);
			return ParseExpr();
		}

		public IEnumerable<SqlExpr> ParseAll(string sql)
		{
			_sql = sql;
			_token.PredicateParse(sql);
			return ParseAllExpr();
		}

		public SqlExpr ParseArithmeticPartial(string sql)
		{
			return ParsePartial(ParseArithmeticExpr, sql);
		}

		private void PredicateParse(string sql)
		{
			_sql = sql;
			_token.PredicateParse(sql);
		}

		public DeclareVariableExpr ParseDeclarePartial(string sql)
		{
			return ParsePartial(ParseDeclare, sql);
		}

		protected DeclareVariableExpr ParseDeclare()
		{
			if (!TryKeyword("DECLARE", out _))
			{
				throw new PrecursorException("DECLARE");
			}
			var varName = ParseVariableName();
			var dataType = ParseDataType();

			SqlExpr defaultValueExpr = null;
			if (TryKeyword("=", out _))
			{
				defaultValueExpr = Any("<Constant>", ParseArithmeticExpr, ParseConstant);
			}

			return new DeclareVariableExpr
			{
				Name = varName,
				DataType = dataType,
				DefaultValue = defaultValueExpr
			};
		}

		protected VariableExpr ParseVariable()
		{
			if (!_token.TryMatch(SqlTokenizer.SqlVariable, out var name))
			{
				throw new PrecursorException("<Variable>");
			}
			return new VariableExpr
			{
				Name = name,
			};
		}

		protected string ParseVariableName()
		{
			if (!_token.TryMatch(SqlTokenizer.SqlVariable, out var name))
			{
				throw new PrecursorException("<Variable>");
			}
			return name;
		}

		protected SqlExpr ParseArithmeticExpr()
		{
			var ops = new string[] { "(", ")", "&", "|", "*", "/", "+", "-" };
			return ParseConcat(() => ParseSubExpr(), ops);
		}

		protected SqlExpr ParseAndOrExpr<T>(Func<T> leftParse)
			where T : SqlExpr
		{
			var ops = new string[] { "(", ")", "AND", "OR" };
			return ParseConcat(() => leftParse(), ops);
		}

		protected SqlExpr ParseAndOrExpr()
		{
			var ops = new string[] { "(", ")", "AND", "OR" };
			return ParseConcat(() => ParseSubExpr(), ops);
		}

		private bool IsOperator(string[] opers)
		{
			//return _token.IsMatchAny(opers.Select(x => Regex.Escape(x)).ToArray());
			return opers.Any(x => _token.IgnoreCase(x));
		}

		protected NegativeExpr ParseNegativeExpr(Func<SqlExpr> readExpr)
		{
			if (!TryKeyword("-", out _))
			{
				throw new PrecursorException("-");
			}

			if (Try(readExpr, out var negativeExpr))
			{
				return new NegativeExpr
				{
					Value = negativeExpr
				};
			}

			throw new Exception();
		}

		protected GroupByExpr ParseGroupBy()
		{
			if (!TryAllKeywords(new[] { "GROUP", "BY" }, out _))
			{
				throw new PrecursorException("GROUP BY");
			}

			var groupExpr = WithComma(() =>
			{
				return ParseSqlIdent();
			});

			return new GroupByExpr
			{
				Items = groupExpr,
			};
		}

		private SqlExpr ParseConcat(Func<SqlExpr> readExpr, string[] opers)
		{
			var operands = new Stack<SqlExpr>();
			var ops = new Stack<string>();
			var readOperand = true;
			while (_token.Text != "")
			{
				if (readOperand && Try(ParseStar, out var starExpr))
				{
					readOperand = false;
					operands.Push(starExpr);
					continue;
				}
				else if (readOperand && Try(() => ParseNegativeExpr(readExpr), out var negativeExpr))
				{
					readOperand = false;
					operands.Push(negativeExpr);
					continue;
				}
				else if (readOperand && !IsOperator(opers) && Try(readExpr, out var expr))
				{
					readOperand = false;
					operands.Push(expr);
					continue;
				}
				else if (!IsOperator(opers))
				{
					break;
				}
				else if (_token.Text == ")" && !ops.Contains("("))
				{
					break;
				}

				readOperand = true;
				if (_token.Text == ")")
				{
					readOperand = false;
				}

				if (!_token.TryIgnoreCase(opers, out var curr_op))
				{
					break;
				}

				if (curr_op == ")")
				{
					ClearStackUnitlPopLeftTerm(operands, ops);
					continue;
				}

				if (ops.Count == 0)
				{
					ops.Push(curr_op);
					continue;
				}

				do
				{
					var stack_op = ops.Peek();
					if (CompareOperPriority(opers, stack_op, curr_op) <= 0)
					{
						ops.Push(curr_op);
						break;
					}

					ClearStack(operands, ops);

					if (ops.Count == 0)
					{
						ops.Push(curr_op);
						break;
					}
				} while (true);
			}

			while (ops.Count > 0)
			{
				ClearStack(operands, ops);
			}

			return operands.Pop();
		}

		private static void ClearStackUnitlPopLeftTerm(Stack<SqlExpr> operands, Stack<string> ops)
		{
			do
			{
				var stack_op = ops.Peek();
				if (stack_op == "(")
				{
					ops.Pop();
					break;
				}
				ClearStack(operands, ops);
			} while (true);
		}

		private static void ClearStack(Stack<SqlExpr> operands, Stack<string> ops)
		{
			var stack_op = ops.Pop();
			var combo = new OperandExpr
			{
				Right = operands.Pop(),
				Oper = stack_op,
				Left = operands.Pop(),
			};
			operands.Push(combo);
		}

		private int CompareOperPriority(string[] opers, string op1, string op2)
		{
			if (op1 == "(" && op2 != "(")
			{
				return -1;
			}
			var opersPriority = opers.Reverse().ToArray();
			var op1Priority = opersPriority.IndexOf(op1);
			var op2Priority = opersPriority.IndexOf(op2);
			if (op1Priority == op2Priority)
			{
				return 0;
			}
			if (op1Priority > op2Priority)
			{
				return 1;
			}
			return -1;
		}

		protected IEnumerable<SqlExpr> ParseAllExpr()
		{
			do
			{
				var expr = ParseExpr();
				yield return expr;
			} while (!string.IsNullOrEmpty(_token.Text));
		}

		public SqlExpr ParseExpr()
		{
			var parseList = new Func<SqlExpr>[]
			{
				ParseCte,
				ParseDeclare,
				ParseSemicolon,
				ParseSelect,
				ParseInsert,
				ParseDelete,
				ParseUpdate,
				ParseSingleLineComment,
				ParseMultiLineComment,
				ParseCreateFunction,
				ParseCreateSp,
				ParseGo,
				ParseSet,
				ParseOnCondition,
				ParseIf,
				ParseWhile,
				ParsePrint,
				ParseUse,
				ParseExec,
				ParseGrant,
				ParseReturn,
				ParseBreak
			};
			for (var i = 0; i < parseList.Length; i++)
			{
				if (Try(parseList[i], out var expr))
				{
					return expr;
				}
			}
			throw new NotSupportedException(GetLastLineCh());
		}

		protected AnyExpr ParseStar()
		{
			if (!TryKeyword("*", out _))
			{
				throw new PrecursorException("*");
			}

			return new AnyExpr();
		}

		public SqlExpr ParseReturnPartial(string sql)
		{
			return ParsePartial(ParseReturn, sql);
		}

		protected SqlExpr ParseReturn()
		{
			if (!TryKeyword("RETURN", out _))
			{
				throw new PrecursorException("RETURN");
			}

			if (TryKeyword("(", out _))
			{
				var innerValueExpr = ParseSubExpr();
				ReadKeyword(")");
				return new GroupExpr
				{
					Expr = new ReturnExpr
					{
						Value = innerValueExpr,
					},
				};
			}

			if (Try(ParseSubExpr, out var valueExpr))
			{
				return new ReturnExpr
				{
					Value = valueExpr
				};
			}

			return new ReturnExpr();
		}

		public SqlExpr ParseGrantPartial(string sql)
		{
			return ParsePartial(ParseGrant, sql);
		}

		protected SqlExpr ParseGrant()
		{
			if (Try(ParseGrantExecuteOn, out var grantExecuteOnExpr))
			{
				return grantExecuteOnExpr;
			}

			if (!_token.TryIgnoreCase("GRANT"))
			{
				throw new PrecursorException("Expect GRANT");
			}

			var permission = ParseIdentWord().Name;
			ReadKeyword("TO");

			var objectId = ParseSqlIdent();
			return new GrantToExpr
			{
				Permission = permission,
				ToObjectId = objectId
			};
		}

		protected ObjectIdExpr ParseObjectId()
		{
			if (!TryAllKeywords(new[] { "OBJECT", "::" }, out _))
			{
				throw new PrecursorException("OBJECT::");
			}

			var name = ParseSqlIdent();

			return new ObjectIdExpr
			{
				Name = name,
			};
		}

		public SqlExpr ParseCasePartial(string sql)
		{
			return ParsePartial(ParseCase, sql);
		}

		protected T ParsePartial<T>(Func<T> parse, string sql)
		{
			_sql = sql;
			_token.PredicateParse(sql);
			return parse();
		}

		protected CaseExpr ParseCase()
		{
			if (!TryKeyword("CASE", out _))
			{
				throw new PrecursorException("CASE");
			}

			TryGet(ParseParenthesesExpr, out var inputExpr);

			var whenList = new List<WhenThenExpr>();
			do
			{
				if (!TryKeyword("WHEN", out _))
				{
					break;
				}
				var conditionExpr = ParseFilterList();
				ReadKeyword("THEN");

				var thenExpr = ParseArithmeticExpr();

				whenList.Add(new WhenThenExpr
				{
					When = conditionExpr,
					Then = thenExpr
				});
			} while (true);


			SqlExpr elseExpr = null;
			if (TryKeyword("ELSE", out _))
			{
				elseExpr = ParseArithmeticExpr();
			}

			ReadKeyword("END");

			return new CaseExpr
			{
				InputExpr = inputExpr,
				WhenList = whenList,
				Else = elseExpr
			};
		}

		protected GrantExecuteOnExpr ParseGrantExecuteOn()
		{
			if (!TryAllKeywords(new[] { "GRANT", "EXECUTE", "ON" }, out var tokens))
			{
				throw new PrecursorException("GRANT EXECUTE ON");
			}

			var objectId = Any("<OBJECT_ID>", ParseObjectId, ParseSqlIdent);

			ReadKeyword("TO");

			var roleId = ParseSqlIdent();

			IdentExpr asDbo = null;
			if (TryKeyword("AS", out _))
			{
				asDbo = ParseSqlIdent1();
			}

			return new GrantExecuteOnExpr
			{
				OnObjectId = objectId,
				ToRoleId = roleId,
				AsDbo = asDbo
			};
		}

		private bool TryKeyword(string keyword, out string token)
		{
			return _token.TryIgnoreCase(keyword, out token);
		}

		private bool TryAnyKeywords(string[] keywords, out string token)
		{
			for (var i = 0; i < keywords.Length; i++)
			{
				if (_token.TryIgnoreCase(keywords[i], out token))
				{
					return true;
				}
			}
			token = null;
			return false;
		}

		private bool TryAllKeywords(string[] keywords, out string[] output)
		{
			var startIndex = _token.CurrentIndex;
			output = new string[keywords.Length];
			for (var i = 0; i < keywords.Length; i++)
			{
				if (!_token.TryIgnoreCase(keywords[i], out output[i]))
				{
					_token.MoveTo(startIndex);
					output[i] = string.Empty;
					return false;
				}
			}
			return true;
		}

		protected SemicolonExpr ParseSemicolon()
		{
			if (!_token.TryIgnoreCase(";"))
			{
				throw new PrecursorException("Expect <;>");
			}
			return new SemicolonExpr();
		}

		protected GoExpr ParseGo()
		{
			if (!_token.TryIgnoreCase("GO"))
			{
				throw new PrecursorException("Expect GO");
			}
			return new GoExpr();
		}

		public SqlExpr ParseSubExpr()
		{
			var parseList = new Func<SqlExpr>[]
			{
				ParseCte,
				ParseCase,
				ParseNot,
				ParseCreateFunction,
				ParseSqlFunc,
				ParseConstant,
				ParseSelect,
				ParseDelete,
				ParseExec,
				ParseParentheses,
				ParseBreak
			};
			for (var i = 0; i < parseList.Length; i++)
			{
				var parse = parseList[i];
				if (Try(parse, out var expr))
				{
					//return ParseCompareOpExpr(ParseInExpr(expr));

					return ParseRightExpr(expr,
						ParseNotLikeExpr,
						ParseInExpr,
						ParseCompareOpExpr);
				}
			}
			throw new Exception(GetLastLineCh() + " Expect sub expr");
		}

		protected SqlExpr ParseRightExpr(SqlExpr leftExpr, params Func<SqlExpr, SqlExpr>[] rightParseList)
		{
			for (var i = 0; i < rightParseList.Length; i++)
			{
				leftExpr = rightParseList[i](leftExpr);
			}
			return leftExpr;
		}

		protected CreateFunctionExpr ParseCreateFunction()
		{
			var startIndex = _token.CurrentIndex;
			if (!TryAllKeywords(new[] { "CREATE", "FUNCTION" }, out _))
			{
				_token.MoveTo(startIndex);
				throw new PrecursorException("CREATE FUNCTION");
			}

			var funcName = ParseSqlIdent();

			ReadKeyword("(");
			//var funcArguments = WithComma(ParseArgumentsList);
			var funcArguments = ParseArgumentsList();
			ReadKeyword(")");
			ReadKeyword("RETURNS");
			var dataType = Any("<Variable DataType> or <DataType>", ParseColumnDataType, ParseDataType);
			ReadKeyword("AS");
			ReadKeyword("BEGIN");
			var body = ParseBody();
			ReadKeyword("END");
			return new CreateFunctionExpr
			{
				Name = funcName,
				Arguments = funcArguments,
				ReturnDataType = dataType,
				Body = body
			};
		}

		protected SqlExpr ParseAnd<T>(Func<T> leftParse)
			where T : SqlExpr
		{
			var ops = new[] { "(", ")", "AND", "OR" };
			return ParseConcat(leftParse, ops);
		}

		protected SqlExpr ParseCompareOp<T>(Func<T> leftParse)
			where T : SqlExpr
		{
			var leftExpr = leftParse();
			if (_token.TryEqual(SqlTokenizer.CompareOps, out var op))
			{
				return new CompareExpr
				{
					Left = leftExpr,
					Oper = op.ToUpper(),
					Right = ParseCompareOp(leftParse)
				};
			}
			return leftExpr;
		}

		protected T ParseIsNull<T>(Func<T> leftParse)
			where T : SqlExpr
		{
			var leftExpr = leftParse();
			return (T)ParseIsNull(leftExpr);
		}

		protected SqlExpr ParseIsNull(SqlExpr left)
		{
			if (!TryKeyword("IS", out _))
			{
				return left;
			}

			if (!TryKeyword("NULL", out var nullToken))
			{
				throw new Exception("<NULL>");
			}

			return new CompareExpr
			{
				Left = left,
				Oper = "IS",
				Right = new NullExpr
				{
					Token = nullToken
				}
			};
		}

		protected NotExpr ParseNot()
		{
			if (!_token.TryIgnoreCase("NOT"))
			{
				throw new PrecursorException("NOT");
			}
			var right = ParseSubExpr();
			return new NotExpr
			{
				Right = right
			};
		}

		protected SqlExpr ParseNotLikeExpr(SqlExpr leftExpr)
		{
			var startIndex = _token.CurrentIndex;
			if (!TryKeyword("NOT", out _))
			{
				return leftExpr;
			}

			if (!TryKeyword("LIKE", out _))
			{
				_token.MoveTo(startIndex);
				return leftExpr;
			}

			var valueExpr = ParseSubExpr();
			return new NotLikeExpr
			{
				Left = leftExpr,
				Right = valueExpr
			};
		}

		public DeleteExpr ParseDeletePartial(string sql)
		{
			return ParsePartial(ParseDelete, sql);
		}

		protected DeleteExpr ParseDelete()
		{
			if (!_token.TryIgnoreCase("DELETE"))
			{
				throw new PrecursorException("DELETE");
			}

			ReadKeyword("FROM");
			var table = ParseSqlIdent();

			var whereExpr = Option(ParseWhere);
			return new DeleteExpr
			{
				Table = table,
				WhereExpr = whereExpr
			};
		}

		public SqlFuncExpr ParseFuncPartial(string sql)
		{
			return ParsePartial(ParseSqlFunc, sql);
		}

		protected CustomFuncExpr ParseCustomFunc()
		{
			var startIndex = _token.CurrentIndex;
			if (!Try(ParseSqlIdent, out var funcName))
			{
				throw new PrecursorException("<Custom Function Name>");
			}

			if (!TryKeyword("(", out _))
			{
				_token.MoveTo(startIndex);
				throw new PrecursorException("(");
			}
			//var subExprList = WithComma(ParseSubExpr);
			var subExprList = WithComma(ParseArithmeticExpr);
			ReadKeyword(")");

			return new CustomFuncExpr
			{
				Name = funcName.Name,
				ObjectId = funcName,
				Arguments = subExprList.Items.ToArray(),
			};
		}

		protected SqlFuncExpr ParseSqlFunc()
		{
			if (Try(ParseCast, out var castExpr))
			{
				return castExpr;
			}

			if (!_token.Try(_token.IsFuncName(out var funcArgsCount), out var funcName))
			{
				if (Try(ParseCustomFunc, out var customFuncExpr))
				{
					return customFuncExpr;
				}

				throw new PrecursorException($"Expect funcname");
			}

			var argsExprs = new List<SqlExpr>();

			ReadKeyword("(");
			for (var i = 0; i < funcArgsCount; i++)
			{
				if (i != 0 && !_token.Try(","))
				{
					Throw("Expect ,");
				}
				//var expr = ParseSubExpr();
				var expr = ParseArithmeticExpr();
				argsExprs.Add(expr);
			}
			ReadKeyword(")");

			return new SqlFuncExpr
			{
				Name = funcName,
				Arguments = argsExprs.ToArray(),
			};
		}

		protected SqlFuncExpr ParseCast()
		{
			if (!_token.TryIgnoreCase("CAST"))
			{
				throw new PrecursorException("CAST");
			}

			ReadKeyword("(");
			var expr = ParseSubExpr();
			ReadKeyword("AS");
			var dataType = ParseDataType();
			ReadKeyword(")");

			var asDataType = new AsDataTypeExpr
			{
				Object = expr,
				DataType = dataType,
			};

			return new SqlFuncExpr
			{
				Name = "CAST",
				Arguments = new SqlExpr[] { asDataType },
			};
		}

		public SqlExpr ParseDataTypePartial(string sql)
		{
			PredicateParse(sql);
			return ParseDataType();
		}

		protected DefineColumnTypeExpr ParseColumnDataType()
		{
			var startIndex = _token.CurrentIndex;
			if (!Try(ParseSqlIdent1, out var columnNameExpr))
			{
				throw new PrecursorException("<ColumnName>");
			}
			if (!Try(ParseDataType, out var dataType))
			{
				_token.MoveTo(startIndex);
				throw new PrecursorException("<DataType>");
			}
			return new DefineColumnTypeExpr
			{
				Name = columnNameExpr,
				DataType = dataType
			};
		}

		protected SqlExpr ParseDataType()
		{
			if (Try(ParseTableType, out var tableTypeExpr))
			{
				return tableTypeExpr;
			}

			if (!_token.TryIgnoreCase(SqlTokenizer.DataTypes, out var dataType))
			{
				throw new PrecursorException("<SqlDataType>");
			}

			var dataSize = Get(ParseDataTypeSize);

			return new DataTypeExpr
			{
				DataType = dataType,
				DataSize = dataSize
			};
		}

		protected TableTypeExpr ParseTableType()
		{
			if (!TryKeyword("TABLE", out var dataType))
			{
				throw new PrecursorException("TABLE");
			}

			ReadKeyword("(");

			var columnTypeList = WithComma(ParseColumnDataType).Items;

			//var columnTypeList = new List<SqlExpr>();
			//do
			//{
			//	if (!TryGet(ParseColumnDataType, out var columnType))
			//	{
			//		break;
			//	}
			//	columnTypeList.Add(columnType);
			//	if (!IsKeyword(","))
			//	{
			//		break;
			//	}
			//} while (true);

			//if (columnTypeList.Count == 0)
			//{
			//	throw new Exception("Must once FieldType");
			//}

			ReadKeyword(")");

			return new TableTypeExpr
			{
				ColumnTypeList = columnTypeList,
			};
		}

		protected DataTypeSizeExpr ParseDataTypeSize()
		{
			if (!_token.Try("("))
			{
				throw new PrecursorException("(");
			}

			var size = ParseInteger().Value;

			int? scaleSize = null;
			if (_token.Try(","))
			{
				scaleSize = ParseInteger().Value;
			}

			ReadKeyword(")");

			return new DataTypeSizeExpr
			{
				Size = size,
				ScaleSize = scaleSize
			};
		}

		protected UseExpr ParseUse()
		{
			if (!_token.TryIgnoreCase("USE"))
			{
				throw new PrecursorException("USE");
			}
			return new UseExpr
			{
				ObjectId = ParseSqlIdent(),
			};
		}

		protected PrintExpr ParsePrint()
		{
			if (!_token.TryIgnoreCase("PRINT"))
			{
				throw new PrecursorException("Expect 'PRINT'");
			}

			var content = ParseConstant();
			return new PrintExpr
			{
				Content = content
			};
		}

		private void ThrowLastLineCh(string message = "")
		{
			throw new Exception(GetLastLineCh() + " " + message);
		}

		private string GetLastLineCh()
		{
			var lnch = _token.GetLineCh(_sql);

			var sb = new StringBuilder();
			sb.AppendLine($"Line:{lnch.LineNumber} Ch:{lnch.ChNumber} CurrToken:'{_token.Text}'");
			sb.AppendLine();

			sb.AppendLine(string.Join("\r\n", lnch.PrevLines));

			var line = lnch.Line.Replace("\t", " ");
			var spaces = new String(' ', line.Length);

			var down = new String('v', _token.Text.Length);
			sb.AppendLine(spaces + down);
			sb.AppendLine(line + $"{_token.Text}");
			var upper = new String('^', _token.Text.Length);
			sb.AppendLine(spaces + upper);
			return sb.ToString();
		}

		protected OnConditionThenExpr ParseOnCondition()
		{
			if (!_token.TryIgnoreCase(":ON"))
			{
				throw new PrecursorException(":ON");
			}

			var condition = ParseIdent().Name;
			var actionName = ParseIdent().Name;
			return new OnConditionThenExpr
			{
				Condition = condition,
				ActionName = actionName
			};
		}

		public SqlExpr ParseSetPartial(string sql)
		{
			return ParsePartial(ParseSet, sql);
		}

		protected SqlExpr ParseSet()
		{
			return Any("<SET xxx>",
				ParseSetVariableEqual,
				ParseSet_Permission_ObjectId_OnOff,
				ParseSet_Options_OnOff,
				ParseSetvar
			);
		}

		protected SetVariableExpr ParseSetVariableEqual()
		{
			var startIndex = _token.CurrentIndex;
			if (!TryKeyword("SET", out _))
			{
				throw new PrecursorException("SET");
			}

			if (!Try(ParseVariableName, out var variableName))
			{
				_token.MoveTo(startIndex);
				throw new PrecursorException("<VariableName>");
			}

			ReadKeyword("=");
			var valueExpr = ParseArithmeticExpr();

			return new SetVariableExpr
			{
				Name = variableName,
				Value = valueExpr
			};
		}

		protected SetBatchVariableExpr ParseSetvar()
		{
			if (!_token.TryIgnoreCase(":setvar"))
			{
				throw new PrecursorException($"Expect ':setvar', but got '{_token.Text}'");
			}

			var name = ParseIdent().Name;
			var value = _token.Read(SqlTokenizer.DoubleQuotedString, nameof(SqlTokenizer.DoubleQuotedString));
			return new SetBatchVariableExpr
			{
				Name = name,
				Value = value
			};
		}

		protected CreateSpExpr ParseCreateSp()
		{
			var startIndex = _token.CurrentIndex;
			if (!_token.TryIgnoreCase("CREATE"))
			{
				throw new PrecursorException("CREATE");
			}

			if (!_token.TryIgnoreCase("PROCEDURE"))
			{
				_token.MoveTo(startIndex);
				throw new PrecursorException("PROCEDURE");
			}

			var spName = ParseSqlIdent();

			var spArgs = ParseArgumentsList();

			ReadKeyword("AS");
			ReadKeyword("BEGIN");
			var body = new List<SqlExpr>();
			do
			{
				var expr = Get(ParseExpr);
				if (expr == null)
				{
					break;
				}
				body.Add(expr);
			} while (true);
			ReadKeyword("END");
			return new CreateSpExpr
			{
				Name = spName,
				Arguments = spArgs,
				Body = body
			};
		}

		private SqlExprList ParseArgumentsList()
		{
			var spArgs = new List<SqlExpr>();
			var mustHaveArg = false;
			do
			{
				if (!_token.TryMatch(SqlTokenizer.SqlVariable, out var argName))
				{
					if (mustHaveArg)
					{
						throw new Exception("argument");
					}
					break;
				}
				var dataType = ParseDataType();

				SqlExpr defaultValue = null;
				if (_token.Try("="))
				{
					defaultValue = ParseConstant();
				}

				spArgs.Add(new ArgumentExpr
				{
					Name = argName,
					DataType = dataType,
					DefaultValue = defaultValue
				});

				if (!_token.Try(","))
				{
					break;
				}

				mustHaveArg = true;
			} while (true);

			return new SqlExprList
			{
				Items = spArgs
			};
		}

		protected SqlExpr ParseInsert()
		{
			var startIndex = _token.CurrentIndex;
			if (!_token.TryIgnoreCase("INSERT"))
			{
				throw new PrecursorException("INSERT");
			}

			var intoToggle = false;
			if (_token.TryIgnoreCase("INTO"))
			{
				intoToggle = true;
			}

			var table = Get(ParseSqlIdent);
			if (table == null)
			{
				_token.MoveTo(startIndex);
				throw new PrecursorException("<table>");
			}

			if (IsKeyword("SELECT"))
			{
				var selectExpr = ParseSelect();
				return new InsertFromSelectExpr
				{
					IntoToggle = intoToggle,
					Table = table,
					FromSelect = selectExpr
				};
			}

			if (!_token.Try("("))
			{
				_token.MoveTo(startIndex);
				throw new PrecursorException("(");
			}
			var fields = WithComma(ParseSqlIdent1);
			ReadKeyword(")");


			ReadKeyword("VALUES");

			var valuesList = new SqlExprList()
			{
				Items = new List<SqlExpr>()
			};

			do
			{
				ReadKeyword("(");
				var values = WithComma(() => Any("Constant or FUNC", ParseSqlFunc, ParseConstant));
				valuesList.Items.Add(values);
				ReadKeyword(")");
				if (!_token.Try(","))
				{
					break;
				}
			} while (true);

			return new InsertExpr
			{
				IntoToggle = intoToggle,
				Table = table,
				Fields = fields,
				ValuesList = valuesList
			};
		}

		protected AssignSetExpr ParseFieldAssignValue()
		{
			var startIndex = _token.CurrentIndex;
			if (!Try(ParseSqlIdent, out var fieldExpr))
			{
				_token.MoveTo(startIndex);
				throw new PrecursorException("<Field>");
			}
			ReadKeyword("=");

			if (Try(ParseArithmeticExpr, out var arithemeticExpr))
			{
				return new AssignSetExpr
				{
					Field = fieldExpr,
					Value = arithemeticExpr
				};
			}

			if (Try(ParseSubExpr, out var subExpr))
			{
				return new AssignSetExpr
				{
					Field = fieldExpr,
					Value = subExpr
				};
			}

			throw new NotSupportedException("Field = xxx Expr");

			//var valueExpr = ParseArithmeticExpr();
			//return new AssignSetExpr
			//{
			//	Field = fieldExpr,
			//	Value = valueExpr
			//};
		}

		public UpdateExpr ParseUpdatePartial(string sql)
		{
			return ParsePartial(ParseUpdate, sql);
		}

		protected UpdateExpr ParseUpdate()
		{
			if (!_token.TryIgnoreCase("UPDATE"))
			{
				throw new PrecursorException("UPDATE");
			}

			var table = ParseSqlIdent();
			ReadKeyword("SET");

			var setFields = WithComma(() => Any("<CASE> or <assign>", ParseCase, ParseFieldAssignValue));

			Try(ParseWhere, out var whereExpr);

			return new UpdateExpr
			{
				Fields = setFields,
				WhereExpr = whereExpr
			};
		}

		protected SetPermissionExpr ParseSet_Permission_ObjectId_OnOff()
		{
			var startIndex = _token.CurrentIndex;
			if (!_token.TryIgnoreCase("SET"))
			{
				throw new PrecursorException("SET");
			}

			if (!_token.TryMatch(RegexPattern.Ident, out var permission))
			{
				_token.MoveTo(startIndex);
				throw new PrecursorException("Expect <Permission>");
			}

			var objectId = Get(ParseSqlIdent);
			if (objectId == null)
			{
				_token.MoveTo(startIndex);
				throw new PrecursorException("Expect <objectId>");
			}

			if (!_token.TryIgnoreCase(new string[] { "ON", "OFF" }, out var toggle))
			{
				_token.MoveTo(startIndex);
				throw new PrecursorException("Expect ON/OFF");
			}
			return new SetPermissionExpr
			{
				Permission = permission,
				ToObjectId = objectId,
				Toggle = string.Equals(toggle, "on", StringComparison.OrdinalIgnoreCase) ? true : false
			};
		}

		private SetOptionsExpr ParseSet_Options_OnOff()
		{
			if (!_token.TryIgnoreCase("SET"))
			{
				throw new PrecursorException($"Expect SET, but got '{_token.Text}'");
			}

			var options = new List<string>();
			var optionExprs = WithComma(ParseIdent);
			options.AddRange(optionExprs.Items.Cast<IdentExpr>().Select(x => x.Name));
			if (options.Count == 0)
			{
				Throw("SET should have option");
			}

			var toggle = ReadAnyKeyword(new[] { "ON", "OFF" });

			return new SetOptionsExpr
			{
				Options = options,
				Toggle = toggle
			};
		}

		protected CommentExpr ParseMultiLineComment()
		{
			if (!_token.TryMatch(SqlTokenizer.MultiLineComment, out var str))
			{
				throw new PrecursorException("/* */");
			}
			return new CommentExpr
			{
				Text = str,
			};
		}

		protected CommentExpr ParseSingleLineComment()
		{
			if (!_token.TryMatch(SqlTokenizer.SingleLineComment, out var token))
			{
				throw new PrecursorException("--");
			}
			return new CommentExpr
			{
				Text = token,
			};
		}

		public SelectExpr ParseSelectPartial(string sql)
		{
			return ParsePartial(ParseSelect, sql);
		}

		protected SqlExpr ParseUnionJoinAll()
		{
			if (TryAllKeywords(new[] { "UNION", "ALL" }, out _))
			{
				return new UnionAllExpr
				{
					Next = ParseSubExpr()
				};
			}

			if (!TryAllKeywords(new[] { "JOIN", "ALL" }, out _))
			{
				throw new PrecursorException("JOIN ALL");
			}
			return new JoinAllExpr
			{
				Next = ParseSubExpr()
			};
		}

		protected SelectExpr ParseSelect()
		{
			if (!_token.TryIgnoreCase("select", out var _))
			{
				throw new PrecursorException();
			}

			var fields = ParseManyColumns();

			Try(ParseFrom, out var fromExpr);

			//var joinTableList = Many(ParseJoin);
			if (!Try(() => Many(ParseInnerJoin), out var joinTableList))
			{
				joinTableList = new SqlExprList()
				{
					Items = new List<SqlExpr>()
				};
			}

			var whereExpr = Get(ParseWhere);

			Try(ParseGroupBy, out var groupByExpr);

			var joinAllList = Many(ParseUnionJoinAll);

			return new SelectExpr
			{
				Fields = fields,
				From = fromExpr,
				Joins = joinTableList.Items,
				WhereExpr = whereExpr,
				GroupByExpr = groupByExpr,
				JoinAllList = joinAllList.Items,
			};
		}

		protected SqlExpr ParseFrom()
		{
			if (!_token.TryIgnoreCase("from", out var _))
			{
				throw new PrecursorException("FROM");
			}

			var fromList = WithComma(() =>
			{
				var sourceExpr = ParseAliasExpr(GetAny(ParseParentheses, ParseSubExpr));
				Try(ParseAlias, out var aliasExpr);
				Try(ParseWithOptions, out var withOptionsExpr);

				return new TableExpr
				{
					Name = sourceExpr,
					AliasName = aliasExpr?.Name,
					WithOptions = withOptionsExpr
				};
			});

			return fromList;

			//return GetAny(ParseSqlFunc, () => WithComma(ParseTableToken));
		}

		private TableExpr ParseTableToken()
		{
			var startIndex = _token.CurrentIndex;
			if (!Try(ParseSqlIdent, out var tableName))
			{
				throw new PrecursorException("<TableName>");
			}

			Try(GetAliasName, out var aliasName);
			Try(ParseWithOptions, out var withOptions);
			return new TableExpr
			{
				Name = tableName,
				AliasName = aliasName,
				WithOptions = withOptions
			};
		}

		private SqlExprList Many(Func<SqlExpr> parse)
		{
			var list = new List<SqlExpr>();
			do
			{
				if (!Try(parse, out var itemExpr))
				{
					break;
				}

				if (itemExpr is SqlExprList itemList)
				{
					return itemList;
				}

				list.Add(itemExpr);
			} while (true);

			return new SqlExprList
			{
				HasComma = false,
				Items = list
			};
		}

		private SqlExprList WithComma<T>(Func<T> parse)
			where T : SqlExpr
		{
			var list = new List<SqlExpr>();
			do
			{
				var item = parse();

				if (item is SqlExprList itemList)
				{
					return itemList;
				}

				list.Add(item);
				if (!_token.Try(","))
				{
					break;
				}
			} while (true);
			return new SqlExprList
			{
				Items = list
			};
		}

		private SqlExprList ParseManyColumns()
		{
			var columns = WithComma(ParseSelectColumn);
			if (columns.Items.Count == 0)
			{
				throw new Exception("field");
			}
			return columns;
		}

		protected SqlExpr ParseEqualExpr(SqlExpr leftExpr)
		{
			if (!TryKeyword("=", out _))
			{
				return leftExpr;
			}

			return new EqualExpr
			{
				Left = leftExpr,
				Right = ParseArithmeticExpr()
			};
		}

		private SqlExpr ParseSelectColumn()
		{
			if (Try(ParseConstant, out var constantExpr))
			{
				return ParseSimpleColumnExpr(ParseEqualExpr(constantExpr));
			}

			//if (_token.IsNumber)
			//{
			//	return ParseInteger();
			//}
			if (_token.IgnoreCase("NOT"))
			{
				return ParseNot();
			}
			if (Try(ParseVariableName, out var variableName))
			{
				ReadKeyword("=");
				return new ColumnSetExpr
				{
					SetVariableName = variableName,
					Column = Any("<SimpleColumn> or <constant>", ParseArithmeticExpr, ParseSimpleColumn, ParseConstant, ParseSubExpr),
				};
			}

			if (Try(ParseArithmeticExpr, out var subExprColumn))
			{
				return ParseSimpleColumnExpr(subExprColumn);
			}

			return ParseSimpleColumn();
		}

		protected BreakExpr ParseBreak()
		{
			if(!TryKeyword(";", out _))
			{
				throw new PrecursorException(";");
			}
			return new BreakExpr();
		}

		public CommonTableExpressionExpr ParseCtePartial(string sql)
		{
			return ParsePartial(ParseCte, sql);
		}

		protected CommonTableExpressionExpr ParseCte()
		{
			var startIndex = _token.CurrentIndex;
			if (!TryKeyword("WITH", out _))
			{
				throw new PrecursorException("WITH");
			}

			if (!Try(ParseIdent, out var cteTableName))
			{
				_token.MoveTo(startIndex);
				throw new PrecursorException("<CTE TableName>");
			}

			var cteColumns = new SqlExprList();
			if (TryKeyword("(", out _))
			{
				cteColumns = WithComma(ParseSqlIdent1);
				ReadKeyword(")");
			}

			ReadKeyword("AS");
			ReadKeyword("(");
			//var select1Expr = ParseSelect();
			//ReadKeyword("UNION");
			//ReadKeyword("ALL");
			//var select2Expr = ParseSelect();

			var innerExpr = ParseSubExpr();

			ReadKeyword(")");

			return new CommonTableExpressionExpr
			{
				TableName = cteTableName,
				Columns = cteColumns,
				InnerExpr = innerExpr,
				//FirstSelect = select1Expr,
				//RecursiveSelect = select2Expr,
			};
		}

		private IdentExpr ParseIdentWord()
		{
			if (!_token.Try(_token.IsIdentWord, out var word))
			{
				throw new Exception($"{_token.Text} should be <Ident Word>");
			}
			return new IdentExpr
			{
				Name = word
			};
		}

		private IdentExpr ParseIdent()
		{
			if (!_token.Try(_token.IsIdent, out var s1))
			{
				throw new PrecursorException("<Ident>");
			}
			return new IdentExpr
			{
				Name = s1
			};
		}

		protected SqlExpr ParseAliasExpr(SqlExpr leftExpr)
		{
			if (!Try(ParseAlias, out var aliasName))
			{
				return leftExpr;
			}
			return new AliasExpr
			{
				Left = leftExpr,
				AliasName = aliasName,
			};
		}

		protected IdentExpr ParseAlias()
		{
			if (Try(ParseIdent, out var aliasName))
			{
				return aliasName;
			}
			if (TryKeyword("AS", out _))
			{
				return ParseIdent();
			}
			throw new PrecursorException("AS <Ident>");
		}

		private string GetAliasName()
		{
			var aliasName = (string)null;
			if (_token.IsIdent)
			{
				aliasName = ParseIdent().Name;
			}
			if (_token.TryIgnoreCase("as"))
			{
				aliasName = ParseIdent().Name;
			}
			return aliasName;
		}

		private IdentExpr ParseSqlIdent1()
		{
			if (!_token.Try(_token.IsSqlIdent, out var str))
			{
				throw new PrecursorException($"<Ident>");
			}
			return new IdentExpr
			{
				Name = str
			};
		}

		private IdentExpr ParseSqlIdent()
		{
			if (!_token.Try(_token.IsSqlIdent, out var s1))
			{
				throw new PrecursorException($"{_token.Text} should be <Ident>");
			}

			if (!_token.Try("."))
			{
				return new IdentExpr
				{
					Name = s1,
				};
			}

			if (!_token.Try(_token.IsSqlIdent, out var s2))
			{
				throw new Exception($"{s1}.<Ident>");
			}

			if (!_token.Try("."))
			{
				return new IdentExpr
				{
					ObjectId = s1,
					Name = s2,
				};
			}

			if (!_token.Try(_token.IsSqlIdent, out var s3))
			{
				throw new Exception($"{s1}.{s2}.<Ident>");
			}

			return new IdentExpr
			{
				DatabaseId = s1,
				ObjectId = s2,
				Name = s3,
			};
		}

		private ColumnExpr ParseSimpleColumnExpr(SqlExpr fieldExpr)
		{
			Try(ParseAlias, out var aliasName);

			return new ColumnExpr
			{
				Name = fieldExpr,
				AliasName = aliasName?.Name,
			};
		}

		private ColumnExpr ParseSimpleColumn()
		{
			if (!Try(ParseSqlIdent, out var identExpr))
			{
				throw new PrecursorException("<Identifier>");
			}

			var aliasName = GetAliasName();

			return new ColumnExpr
			{
				Name = identExpr,
				AliasName = aliasName,
			};
		}

		protected Hex16NumberExpr ParseHex16Number()
		{
			if (!_token.TryMatch(SqlTokenizer.Hex16Number, out var hex))
			{
				throw new PrecursorException("<HEX16>");
			}

			return new Hex16NumberExpr
			{
				Value = hex
			};
		}

		public IntegerExpr ParseInteger()
		{
			if (!_token.TryInteger(out var intValue))
			{
				throw new PrecursorException($"{_token.Text} should be integer");
			}

			return new IntegerExpr
			{
				Value = intValue
			};
		}

		public WithOptionsExpr ParseWithOptions()
		{
			if (!_token.TryIgnoreCase("with"))
			{
				throw new PrecursorException("WITH");
			}

			if (!_token.Try("("))
			{
				throw new Exception("(");
			}

			var withOptions = new List<string>();
			if (_token.TryIgnoreCase("nolock", out var nolockToken))
			{
				withOptions.Add(nolockToken);
			}

			if (!_token.Try(")"))
			{
				throw new Exception(")");
			}

			return new WithOptionsExpr
			{
				Options = withOptions
			};
		}

		public SqlExpr ParseWherePartial(string sql)
		{
			return ParsePartial(ParseWhere, sql);
		}

		protected SqlExpr ParseWhere()
		{
			if (!_token.TryIgnoreCase("where"))
			{
				throw new PrecursorException("WHERE");
			}
			return ParseFilterList();
		}

		public SqlExpr ParseIfPartial(string sql)
		{
			return ParsePartial(ParseIf, sql);
		}


		protected SqlExpr ParseCompareOpExpr(SqlExpr leftExpr)
		{
			if (!_token.TryEqual(SqlTokenizer.CompareOps, out var op))
			{
				return leftExpr;
			}

			return new CompareExpr
			{
				Left = leftExpr,
				Oper = op.ToUpper(),
				Right = ParseSubExpr()
			};
		}

		protected SqlExpr ParseInExpr(SqlExpr leftExpr)
		{
			if (!TryKeyword("IN", out _))
			{
				return leftExpr;
			}

			ReadKeyword("(");
			var values = WithComma(ParseSubExpr);
			ReadKeyword(")");

			return new InExpr
			{
				LeftExpr = leftExpr,
				Values = values
			};
		}

		public WhileExpr ParseWhilePartial(string sql)
		{
			return ParsePartial(ParseWhile, sql);
		}

		protected WhileExpr ParseWhile()
		{
			if (!TryKeyword("WHILE", out _))
			{
				throw new PrecursorException("WHILE");
			}

			var booleanExpr = ParseArithmeticExpr();
			ReadKeyword("BEGIN");
			var body = ParseBody();
			ReadKeyword("END");

			return new WhileExpr
			{
				Condition = booleanExpr,
				Body = body
			};
		}

		protected SqlExpr ParseIf()
		{
			if (!TryKeyword("IF", out _))
			{
				throw new PrecursorException("IF");
			}
			var filter = ParseFilterList();
			ReadKeyword("BEGIN");
			var body = ParseBody();
			ReadKeyword("END");

			List<SqlExpr> elseBody = new List<SqlExpr>();
			if (TryKeyword("ELSE", out _))
			{
				ReadKeyword("BEGIN");
				elseBody = ParseBody();
				ReadKeyword("END");
			}

			return new IfExpr
			{
				Condition = filter,
				Body = body,
				ElseBody = elseBody
			};
		}

		private List<SqlExpr> ParseBody()
		{
			var body = new List<SqlExpr>();
			do
			{
				var expr = Get(ParseExpr);
				if (expr == null)
				{
					break;
				}
				body.Add(expr);
			} while (true);
			return body;
		}

		private SqlExpr ParseFilterList()
		{
			var postfixExprs = new List<object>();
			var ops = new Stack<string>();
			do
			{
				//var filter = ParseFilter();
				//var filter = ParseCompareOpExpr(ParseArithmeticExpr());
				var filter = ParseParenthesesExpr();
				postfixExprs.Add(filter);
				if (!_token.TryIgnoreCase(new[] { "AND", "OR" }, out var op))
				{
					break;
				}
				if (ops.Count >= 1)
				{
					var prev_op = ops.Pop();
					if (CompareOp(prev_op, op) > 0)
					{
						ops.Push(prev_op);
						postfixExprs.Add(op);
					}
					else
					{
						ops.Push(op);
						postfixExprs.Add(prev_op);
					}
				}
				else
				{
					ops.Push(op);
				}
			} while (true);

			if (ops.Count > 0)
			{
				postfixExprs.Add(ops.Pop());
			}

			return ComputePostfixExprs(postfixExprs);
		}

		private SqlExpr ComputePostfixExprs(List<object> postfixExprs)
		{
			var st = new Stack<SqlExpr>();
			var iter = postfixExprs.GetEnumerator();
			while (iter.MoveNext())
			{
				var item = iter.Current;
				if (item is SqlExpr sqlExpr)
				{
					st.Push(sqlExpr);
				}
				else
				{
					var oper = item as string;
					var right = st.Pop();
					var left = st.Pop();
					var expr = new OperandExpr
					{
						Left = left,
						Oper = oper,
						Right = right,
					};
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
			if (op1.Equals(op2))
			{
				return 0;
			}
			if (op1 == "AND" && op2 == "OR")
			{
				return 1;
			}
			return -1;
		}

		protected NullExpr ParseNull()
		{
			if (!_token.TryIgnoreCase("NULL", out var token))
			{
				throw new PrecursorException("NULL");
			}
			return new NullExpr
			{
				Token = token
			};
		}

		protected DecimalExpr ParseDecimal()
		{
			if (!_token.TryMatch(SqlTokenizer.DecimalNumber, out var decimalStr))
			{
				throw new PrecursorException("<Float>");
			}
			var decimalValue = Decimal.Parse(decimalStr);
			return new DecimalExpr
			{
				Value = decimalValue,
			};
		}

		protected NegativeExpr ParseNegativeNumber()
		{
			var startIndex = _token.CurrentIndex;
			if (!TryKeyword("-", out _))
			{
				throw new PrecursorException("-");
			}

			if (!TryGetAny(out var numberExp, ParseHex16Number, ParseDecimal, ParseInteger))
			{
				_token.MoveTo(startIndex);
				throw new PrecursorException("<Number>");
			}

			return new NegativeExpr
			{
				Value = numberExp
			};
		}

		private SqlExpr ParseConstant()
		{
			var expr = GetAny(
				ParseNull,
				ParseNegativeNumber,
				ParseHex16Number,
				ParseDecimal,
				ParseInteger,
				ParseString,
				ParseSqlIdent,
				ParseVariable,
				ParseStar);
			if (expr == null)
			{
				throw new PrecursorException("<Constant>");
			}
			return expr;
		}

		private StringExpr ParseString()
		{
			if (!_token.Try(_token.IsSqlString, out var str))
			{
				throw new PrecursorException("Expect string");
			}
			return new StringExpr
			{
				Text = str,
			};
		}

		public SqlExpr ParseFilterPartial(string sql)
		{
			PredicateParse(sql);
			return ParseFilter();
		}

		private SqlExpr ParseWithCompareOp(Func<SqlExpr> leftParse)
		{
			var leftExpr = leftParse();

			var likeExpr = Get(ParseLike, leftExpr);
			if (likeExpr != null)
			{
				return likeExpr;
			}

			var compareExpr = Get(ParseCompareOp, leftExpr, leftParse);
			if (compareExpr != null)
			{
				return compareExpr;
			}

			return leftExpr;
		}

		public SqlExpr ParseEqualOpPartial(string sql)
		{
			return ParsePartial(ParseSubExpr, sql);
			//return ParsePartial(() => ParseEqualOp(ParseSubExpr()), sql);
		}

		protected SqlExpr ParseEqualOp(SqlExpr leftExpr)
		{
			if (!TryKeyword("=", out _))
			{
				throw new PrecursorException("=");
			}

			return new OperandExpr
			{
				Left = leftExpr,
				Oper = "=",
				Right = ParseSubExpr()
			};
		}

		private SqlExpr ParseFilter()
		{
			if (IsKeyword("("))
			{
				return ParseParentheses();
			}

			if (_token.IgnoreCase("NOT"))
			{
				return ParseNot();
			}

			var left = ParseArithmeticExpr();
			return left;
		}

		protected ExecuteExpr ParseExec()
		{
			if (!_token.TryIgnoreCase(new[] { "EXECUTE", "EXEC" }, out var execStr))
			{
				throw new PrecursorException("Expect EXEC");
			}

			var method = ParseSqlIdent();
			var arguments = new List<SqlExpr>();

			var first = true;
			do
			{
				var sqlParam = GetAny(ParseParameterNameAssign, ParseConstant);
				if (sqlParam != null)
				{
					arguments.Add(sqlParam);
				}

				if (first && sqlParam == null)
				{
					break;
				}
				else if (!first && sqlParam == null)
				{
					throw new Exception("ParseExec Expect param");
				}
				first = false;
				if (!_token.Try(","))
				{
					break;
				}
			} while (true);

			return new ExecuteExpr
			{
				ExecName = execStr,
				Method = method,
				Arguments = arguments.ToArray(),
			};
		}

		private T Option<T>(Func<T> parse)
		{
			try
			{
				return parse();
			}
			catch (PrecursorException)
			{
				return default(T);
			}
		}

		private T Get<T>(Func<T> parse)
		{
			try
			{
				return parse();
			}
			catch
			{
				return default(T);
			}
		}

		private T Get<T>(Func<SqlExpr, T> parse, SqlExpr left)
		{
			try
			{
				return parse(left);
			}
			catch (PrecursorException)
			{
				return default(T);
			}
		}

		private T Get<T>(Func<SqlExpr, Func<SqlExpr>, T> parse, SqlExpr left, Func<SqlExpr> rightParse)
		{
			try
			{
				return parse(left, rightParse);
			}
			catch (PrecursorException)
			{
				return default(T);
			}
		}

		private bool Try<T>(Func<T> parse, out T output)
		{
			try
			{
				if (_token.Text == String.Empty)
				{
					throw new PrecursorException();
				}

				output = parse();
				return true;
			}
			catch (PrecursorException)
			{
				output = default(T);
				return false;
			}
		}


		private bool TryGet<T>(Func<T> parse, out T output)
		{
			output = Get<T>(parse);
			return output != null;
		}


		private bool TryGetAny(out SqlExpr output, params Func<SqlExpr>[] parseList)
		{
			for (var i = 0; i < parseList.Length; i++)
			{
				if (Try(parseList[i], out output))
				{
					return true;
				}
			}
			output = default;
			return false;
		}

		private SqlExpr Any(string expect, params Func<SqlExpr>[] parseList)
		{
			for (var i = 0; i < parseList.Length; i++)
			{
				var parse = parseList[i];
				try
				{
					return parse();
				}
				catch (PrecursorException)
				{
					continue;
				}
			}
			throw new PrecursorException(expect);
		}

		private SqlExpr EatAny(params Func<SqlExpr>[] parseList)
		{
			for (var i = 0; i < parseList.Length; i++)
			{
				var parse = parseList[i];
				try
				{
					return parse();
				}
				catch (PrecursorException)
				{
					continue;
				}
			}
			throw new Exception();
		}

		private SqlExpr GetAny(params Func<SqlExpr>[] parseList)
		{
			for (var i = 0; i < parseList.Length; i++)
			{
				var parse = parseList[i];
				try
				{
					return parse();
				}
				catch (PrecursorException)
				{
					continue;
				}
			}
			return default(SqlExpr);
		}

		private CompareExpr ParseCompareOp(SqlExpr left, Func<SqlExpr> parseRight)
		{
			var op = string.Empty;
			CompareExpr parseAction()
			{
				SqlExpr right = null;
				if (op.IsSql("in"))
				{
					right = ParseParentheses();
				}
				else
				{
					//right = ParseSubExpr();
					right = parseRight();
				}
				return new CompareExpr
				{
					Left = left,
					Oper = op,
					Right = right,
				};
			}

			if (_token.TryIgnoreCase("NOT", out var op1) && _token.TryIgnoreCase("LIKE", out var op2))
			{
				op = $"{op1} {op2}";
				return parseAction();
			}

			if (!_token.Try(_token.IsCompareOp, out op))
			{
				throw new PrecursorException("<compare oper>");
			}
			return parseAction();
		}

		protected SqlExpr ParseParenthesesExpr()
		{
			if (!TryKeyword("(", out _))
			{
				return ParseWithCompareOp(ParseArithmeticExpr);
			}
			var subExpr = ParseWithCompareOp(ParseArithmeticExpr);
			ReadKeyword(")");


			var groupExpr = new GroupExpr
			{
				Expr = subExpr,
			};

			return ParseCompareOpExpr(groupExpr);
		}

		protected SqlExpr WithParentheses(Func<SqlExpr> parse)
		{
			if (!_token.Try("("))
			{
				return parse();
			}
			var innerExpr = parse();
			ReadKeyword(")");
			return new GroupExpr
			{
				Expr = innerExpr,
			};
		}

		protected GroupExpr ParseParentheses()
		{
			if (!_token.Try("("))
			{
				throw new PrecursorException("(");
			}
			//var subExpr = ParseWithCompareOp(ParseSubExpr);
			var subExpr = ParseWithCompareOp(ParseArithmeticExpr);
			ReadKeyword(")");

			return new GroupExpr
			{
				Expr = subExpr,
			};
		}

		private bool IsKeyword(string keyword)
		{
			return _token.IgnoreCase(keyword);
		}

		private bool IsAnyKeyword(params string[] keywords)
		{
			return keywords.Any(keyword => IsKeyword(keyword));
		}

		private string ReadKeyword(string keyword)
		{
			if (!_token.TryIgnoreCase(keyword, out var token))
			{
				ThrowLastLineCh($"Expect is '{keyword.ToUpper()}'");
			}
			return token;
		}

		private bool ReadKeywordOption(string keyword)
		{
			return _token.TryIgnoreCase(keyword);
		}

		private bool ReadKeywordOption(string keyword, out string token)
		{
			return _token.TryIgnoreCase(keyword, out token);
		}

		private bool ReadKeywordOption(string[] keywords, out string token)
		{
			foreach (var keyword in keywords)
			{
				var success = _token.TryIgnoreCase(keyword, out token);
				if (success)
				{
					return true;
				}
			}
			token = null;
			return false;
		}


		private string ReadAnyKeyword(string[] keywords)
		{
			var token = string.Empty;
			foreach (var keyword in keywords)
			{
				var success = _token.TryIgnoreCase(keyword, out token);
				if (success)
				{
					return token;
				}
			}
			var msg = string.Join(" ", keywords);
			throw new Exception(CreateThrowMessage($"Expect any keyword in {msg}, but got {_token.Text}"));
		}

		private JoinExpr ParseInnerJoin()
		{
			var startIndex = _token.CurrentIndex;
			ReadKeywordOption(new[] { "inner", "left", "right" }, out var joinTypeStr);

			var outerToken = string.Empty;
			if (!string.IsNullOrEmpty(joinTypeStr))
			{
				TryKeyword("OUTER", out outerToken);
			}

			if (startIndex == _token.CurrentIndex && !IsKeyword("JOIN"))
			{
				throw new PrecursorException("JOIN");
			}

			ReadKeyword("JOIN");

			if (IsKeyword("ALL"))
			{
				_token.MoveTo(startIndex);
				throw new PrecursorException("<TABLE>");
			}

			var table = ParseTableToken();

			ReadKeyword("on");

			var filterList = ParseFilterList();

			var joinType = (JoinType)Enum.Parse(typeof(JoinType), joinTypeStr, true);
			return new JoinExpr
			{
				Table = table,
				JoinType = joinType,
				OuterToken = outerToken,
				Filter = filterList
			};
		}

		public SqlExpr ParseParameterNameAssign()
		{
			if (!_token.TryMatch(SqlTokenizer.SqlVariable, out var varName))
			{
				throw new PrecursorException("Expect SqlVariable");
			}

			if (!_token.Try("="))
			{
				return new SqlVariableExpr
				{
					Name = varName
				};
			}

			var value = ParseConstant();
			return new SpParameterExpr
			{
				Name = varName,
				Value = value
			};
		}

		public LikeExpr ParseLike(SqlExpr left)
		{
			if (!_token.TryIgnoreCase("like"))
			{
				throw new PrecursorException("LIKE");
			}

			if (!_token.Try(_token.IsSqlString, out var likeStr))
			{
				throw new Exception($"{_token.Text} should is string");
			}

			return new LikeExpr
			{
				Left = left,
				Right = likeStr
			};
		}

		private string CreateThrowMessage(string message)
		{
			var info = "";
			if (_token.Curr != null)
			{
				info = $"index:{_token.Curr.Index}, curr:'{_token.Text}'";
			}
			return $"{info} {message}.";
		}

		private void Throw(string message)
		{
			throw new Exception(CreateThrowMessage(message));
		}

		private void Initialize()
		{
			var arithmeticOptions = new InfixToPostfixOptions<SqlExpr>()
			{
				Operators = new[] { "*", "/", "+", "-" },
				CreateBinaryOperator = (left, op, right) => new ArithmeticExpr
				{
					Left = left,
					Oper = op,
					Right = right
				},
				GetCurrentTokenText = () => _token.Text,
				NextToken = () => _token.Move(),
				ReadToken = () => ParseSubExpr(),
			};
			_arithmetic = new InfixToPostfix<SqlExpr>(arithmeticOptions);
		}
	}

	public enum JoinType
	{
		Inner,
		Left,
		Right
	}

	public static class Enumer
	{
		public static T Next<T>(this IEnumerator<T> enumerator)
		{
			enumerator.MoveNext();
			return enumerator.Current;
		}
	}
}
