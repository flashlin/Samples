using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using PreviewLibrary.Helpers;
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

		public SqlExpr ParseDeclarePartial(string sql)
		{
			return ParsePartial(ParseDeclare, sql);
		}

		protected SqlExprList AtLeast(Func<SqlExpr> parse)
		{
			var list = Many(parse);
			if (list.Items.Count == 0)
			{
				throw new Exception();
			}
			return list;
		}

		protected SqlExpr ParseDeclare()
		{
			if (!TryKeyword("DECLARE", out _))
			{
				throw new PrecursorException("DECLARE");
			}

			DeclareVariableExpr variableDataType()
			{
				var varName = ParseVariableName();

				TryKeyword("AS", out _);
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

			var listExpr = WithComma(variableDataType);
			if (listExpr.Items.Count == 0)
			{
				throw new Exception("<Variable> <DataType>");
			}

			listExpr.HasComma = false;
			return listExpr;
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

		protected SqlExpr ParseElement()
		{

			return Any("", ParseConstant);
		}

		protected SqlExpr ParseArithmeticExpr(Func<SqlExpr> innerParse)
		{
			//var ops = new string[] { "(", ")", "&", "|", "*", "/", "+", "-" };
			var ops = new string[] { "(", ")", "AND", "OR",
				"LIKE", "NOT LIKE", "NOT IN",
				"&", "|", "*", "/", "+", "-",
				"<>", ">=", "<", ">", "=",
				};
			return ParseConcat(() => innerParse(), ops);
		}

		protected SqlExpr ParseArithmeticExpr()
		{
			return ParseArithmeticExpr(ParseSubExpr);
		}

		private bool IsOperator(string[] opers)
		{
			return opers.Any(x => _token.IgnoreCase(x));
		}

		protected NegativeExpr ParseNegativeExpr(Func<SqlExpr> readExpr)
		{
			if (!TryKeyword("-", out _))
			{
				throw new PrecursorException("-");
			}

			if (TryGet(readExpr, out var negativeExpr))
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

			if (IsKeyword(")"))
			{
				throw new PrecursorException("(");
			}

			while (_token.Text != "")
			{
				if (readOperand && TryGet(ParseStar, out var starExpr))
				{
					readOperand = false;
					operands.Push(starExpr);
					continue;
				}
				else if (readOperand && TryGet(() => ParseNegativeExpr(readExpr), out var negativeExpr))
				{
					readOperand = false;
					operands.Push(negativeExpr);
					continue;
				}
				else if (readOperand && !IsOperator(opers) && TryGet(readExpr, out var expr))
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

			if (operands.Count == 0)
			{
				throw new PrecursorException();
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
					var expr = operands.Pop();
					operands.Push(new GroupExpr { InnerExpr = expr });
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
			//operands.Push(new GroupExpr{ Expr = combo });
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
				ParseBegin,
				ParseCreatePartitionFunction,
				ParseCreatePartitionScheme,
				ParseAlter,
				ParseWaitforDelay,
				ParseMerge,
				ParseCte,
				ParseDeclare,
				ParseSemicolon,
				ParseSelect,
				ParseMergeInsert,
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
				ParseBreak,
				ParseCommit,
			};
			for (var i = 0; i < parseList.Length; i++)
			{
				if (TryGet(parseList[i], out var expr))
				{
					return expr;
				}
			}
			if (string.IsNullOrEmpty(_token.Text))
			{
				return default;
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
				var innerValueExpr = ParseArithmeticExpr();
				ReadKeyword(")");
				return new GroupExpr
				{
					InnerExpr = new ReturnExpr
					{
						Value = innerValueExpr,
					},
				};
			}

			if (TryGet(ParseFunctionWithParentheses, out var funcExpr))
			{
				return new ReturnExpr
				{
					Value = funcExpr
				};
			}

			if (TryGet(() => ParseArithmeticExpr(ParseConstantForReturn), out var valueExpr))
			{
				return new ReturnExpr
				{
					Value = valueExpr
				};
			}

			if (TryGet(ParseCase, out var caseExpr))
			{
				return new ReturnExpr
				{
					Value = caseExpr
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
			if (TryGet(ParseGrantExecuteOn, out var grantExecuteOnExpr))
			{
				return grantExecuteOnExpr;
			}

			if (TryGet(ParseGrantExecOn, out var grantExecOnExpr))
			{
				return grantExecOnExpr;
			}

			if (!_token.TryIgnoreCase("GRANT"))
			{
				throw new PrecursorException("Expect GRANT");
			}

			var permission = ParseIdentWord().Name;
			ReadKeyword("TO");

			var objectIds = WithComma(ParseSqlIdent);
			return new GrantToExpr
			{
				Permission = permission,
				ToObjectIds = objectIds
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

			SqlExpr inputExpr = null;
			if (!IsKeyword("WHEN"))
			{
				TryGet(ParseParenthesesExpr, out inputExpr);
			}

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
			if (!TryAllKeywords(new[] { "GRANT", "EXECUTE", "ON" }, out _))
			{
				throw new PrecursorException("GRANT EXECUTE ON");
			}

			var objectId = Any("<OBJECT_ID>", ParseObjectId, ParseSqlIdent);

			ReadKeyword("TO");

			var roleIds = WithComma(ParseSqlIdent);

			IdentExpr asDbo = null;
			if (TryKeyword("AS", out _))
			{
				asDbo = ParseSqlIdent1();
			}

			return new GrantExecuteOnExpr
			{
				ExecAction = "EXECUTE",
				OnObjectId = objectId,
				ToRoleIds = roleIds,
				AsDbo = asDbo
			};
		}

		protected GrantExecuteOnExpr ParseGrantExecOn()
		{
			if (!TryAllKeywords(new[] { "GRANT", "EXEC", "ON" }, out _))
			{
				throw new PrecursorException("GRANT EXEC ON");
			}

			var objectId = Any("<OBJECT_ID>", ParseObjectId, ParseSqlIdent);

			ReadKeyword("TO");

			var roleIds = WithComma(ParseSqlIdent);

			IdentExpr asDbo = null;
			if (TryKeyword("AS", out _))
			{
				asDbo = ParseSqlIdent1();
			}

			return new GrantExecuteOnExpr
			{
				ExecAction = "EXEC",
				OnObjectId = objectId,
				ToRoleIds = roleIds,
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


		private Func<string[]> Keywords(params string[] keywords)
		{
			return () =>
			{
				var startIndex = _token.CurrentIndex;
				var output = new string[keywords.Length];
				for (var i = 0; i < keywords.Length; i++)
				{
					if (!_token.TryIgnoreCase(keywords[i], out output[i]))
					{
						_token.MoveTo(startIndex);
						output[i] = string.Empty;
						throw new PrecursorException(string.Join(" ", keywords));
					}
				}
				return output;
			};
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


		private bool TryAllKeywords(string keywords, out string[] output)
		{
			var condense = keywords.CondenseSpaces();
			var keywordsList = condense.Split(' ');
			return TryAllKeywords(keywordsList, out output);
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
				ParseDistinct,
				ParseCommit,
				ParseRankOver,
				ParseMergeInsert,
				ParseWaitforDelay,
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
				ParseBreak,
				ParseIf,
			};
			for (var i = 0; i < parseList.Length; i++)
			{
				var parse = parseList[i];
				if (TryGet(parse, out var expr))
				{
					return ParseRightExpr(expr,
						ParseBetweenExpr,
						ParseNotLikeExpr,
						ParseNotInExpr,
						ParseInExpr,
						ParseCompareOpExpr);
				}
			}
			throw new NotSupportedException(GetLastLineCh() + " Expect sub expr");
		}

		protected SqlExpr ParseRightExpr(SqlExpr leftExpr, params Func<SqlExpr, SqlExpr>[] rightParseList)
		{
			for (var i = 0; i < rightParseList.Length; i++)
			{
				leftExpr = rightParseList[i](leftExpr);
			}
			return leftExpr;
		}

		protected CreatePartitionSchemeExpr ParseCreatePartitionScheme()
		{
			if (!TryGet(Keywords("CREATE", "PARTITION", "SCHEME"), out _))
			{
				throw new PrecursorException("CREATE PARITION SCHEME");
			}

			var partitionSchemeName = ParseSqlIdent();
			if (!TryGet(Keywords("AS", "PARTITION"), out _))
			{
				throw new ParseException("AS PARITION");
			}
			var partitionFunctionName = ParseSqlIdent();

			TryKeyword("ALL", out var allExpr);
			ReadKeyword("TO");
			ReadKeyword("(");
			var fileGroupNameList = WithComma(ParseSqlIdent1);
			ReadKeyword(")");

			return new CreatePartitionSchemeExpr
			{
				SchemeName = partitionSchemeName,
				FunctionName = partitionFunctionName,
				All = allExpr,
				FileGroupList = fileGroupNameList,
			};
		}

		protected CreatePartitionFunctionExpr ParseCreatePartitionFunction()
		{
			if (!TryAllKeywords(new[] { "CREATE", "PARTITION", "FUNCTION" }, out _))
			{
				throw new PrecursorException("CREATE PARTITION FUNCTION");
			}

			var partitionFunctionName = ParseSqlIdent();
			ReadKeyword("(");
			var inputParameterType = ParseDataType();
			ReadKeyword(")");
			if (!TryAllKeywords(new[] { "AS", "RANGE" }, out _))
			{
				throw new ParseException("AS RANGE");
			}

			if (!TryAllKeywords(new[] { "FOR", "VALUES" }, out _))
			{
				throw new ParseException("FOR VALUES");
			}
			ReadKeyword("(");
			var boundaryValueList = WithComma(ParseConstant);
			ReadKeyword(")");

			return new CreatePartitionFunctionExpr
			{
				FuncName = partitionFunctionName,
				InputParameterType = inputParameterType,
				BoundaryValueList = boundaryValueList
			};
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

			var right = ParseArithmeticExpr();
			return new NotExpr
			{
				Right = right
			};
		}

		protected SqlExpr ParseNotLikeExpr(SqlExpr leftExpr)
		{
			if (!TryKeyword("NOT LIKE", out _))
			{
				return leftExpr;
			}

			var valueExpr = ParseSubExpr();
			return new NotLikeExpr
			{
				Left = leftExpr,
				Right = valueExpr
			};
		}

		protected SqlExpr ParseNotInExpr(SqlExpr leftExpr)
		{
			var startIndex = _token.CurrentIndex;
			if (!TryKeyword("NOT IN", out _))
			{
				return leftExpr;
			}

			ReadKeyword("(");
			var valueExpr = WithComma(ParseSubExpr);
			ReadKeyword(")");

			return new NotInExpr
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

			TryGet(ParseTop, out var topExpr);

			TryKeyword("FROM", out _);
			var table = ParseSqlIdent();

			TryGet(ParseWithOptions, out var withOptionsExpr);

			TryGet(ParseOutput, out var outputExpr);
			TryGet(ParseInto, out var intoExpr);

			var whereExpr = Option(ParseWhere);
			return new DeleteExpr
			{
				Top = topExpr,
				Table = table,
				WithOptions = withOptionsExpr,
				OutputExpr = outputExpr,
				IntoExpr = intoExpr,
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
			if (!TryGet(ParseSqlIdent, out var funcName))
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
			if (TryGet(ParseCast, out var castExpr))
			{
				return castExpr;
			}

			if (!_token.Try(_token.IsFuncName(out var funcArgsCount), out var funcName))
			{
				if (TryGet(ParseCustomFunc, out var customFuncExpr))
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
			//var expr = ParseSubExpr();
			var expr = ParseArithmeticExpr();

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
			return ParsePartial(ParseDataType, sql);
		}

		protected DefineColumnTypeExpr ParseColumnDataType()
		{
			var startIndex = _token.CurrentIndex;
			if (!TryGet(ParseSqlIdent1, out var columnNameExpr))
			{
				throw new PrecursorException("<ColumnName>");
			}
			if (!TryGet(ParseDataType, out var dataType))
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

		protected WhenMatchedExpr ParseWhenMatched()
		{
			if (!TryAllKeywords(new[] { "WHEN", "MATCHED" }, out _))
			{
				throw new PrecursorException("WHEN MATCHED");
			}
			TryGet(ParseFilterList, out var searchCondition);
			ReadKeyword("THEN");
			var body = ParseBody();

			return new WhenMatchedExpr
			{
				Condition = searchCondition,
				Body = body
			};
		}

		protected WhenNotMatchedExpr ParseWhenNotMatchedByTarget()
		{
			if (!TryAllKeywords(new[] { "WHEN", "NOT", "MATCHED" }, out _))
			{
				throw new PrecursorException("WHEN NOT MATCHED");
			}

			TryAllKeywords(new[] { "BY", "TARGET" }, out var byTargetExpr);
			var byTargetToken = string.Join(" ", byTargetExpr);

			SqlExpr searchCondition = null;
			if (!IsKeyword("THEN"))
			{
				TryGet(ParseFilterList, out searchCondition);
			}
			ReadKeyword("THEN");
			var body = ParseBody();

			return new WhenNotMatchedExpr
			{
				Condition = searchCondition,
				ByToken = byTargetToken,
				Body = body
			};
		}


		protected WhenNotMatchedExpr ParseWhenNotMatchedBySource()
		{
			if (!TryAllKeywords(new[] { "WHEN", "NOT", "MATCHED" }, out _))
			{
				throw new PrecursorException("WHEN NOT MATCHED");
			}

			ReadKeyword("BY");
			ReadKeyword("SOURCE");

			TryGet(ParseFilterList, out var searchCondition);
			ReadKeyword("THEN");
			var body = ParseBody();

			return new WhenNotMatchedExpr
			{
				Condition = searchCondition,
				ByToken = "BY SOURCE",
				Body = body
			};
		}

		protected MergeExpr ParseMerge()
		{
			if (!TryKeyword("MERGE", out _))
			{
				throw new PrecursorException("MERGE");
			}
			TryKeyword("INTO", out var intoToken);

			var targetTable = ParseSubExpr();
			TryGet(ParseAlias, out var targetAlias);

			ReadKeyword("USING");
			var sourceTable = ParseSubExpr();
			TryGet(ParseAlias, out var sourceAlias);

			ReadKeyword("ON");
			var onFilterList = ParseFilterList();

			TryGet(ParseWhenMatched, out var whenMatchedExpr);
			TryGet(ParseWhenNotMatchedByTarget, out var whenNotMatchedByTargetExpr);
			TryGet(ParseWhenNotMatchedBySource, out var whenNotMatchedBySourceExpr);

			return new MergeExpr
			{
				IntoToken = intoToken,
				TargetTable = targetTable,
				TargetAlias = targetAlias,
				SourceTable = sourceTable,
				SourceAlias = sourceAlias,
				OnCondition = onFilterList,
				WhenMatched = whenMatchedExpr,
				WhenNotMatched = whenNotMatchedByTargetExpr,
				WhenNotMatchedBySource = whenNotMatchedBySourceExpr,
			};
		}

		protected TvpTableTypeExpr ParseTvpTable()
		{
			var startIndex = _token.CurrentIndex;
			if (!TryGet(ParseSqlIdent, out var tvpName))
			{
				throw new PrecursorException("<TVP TYPE>");
			}

			if (!TryKeyword("READONLY", out _))
			{
				_token.MoveTo(startIndex);
				throw new PrecursorException("READONLY");
			}

			return new TvpTableTypeExpr
			{
				Name = tvpName,
			};
		}

		protected SqlExpr ParseDataType()
		{
			if (TryGet(ParseTableType, out var tableTypeExpr))
			{
				return tableTypeExpr;
			}

			if (TryGet(ParseTvpTable, out var tvpName))
			{
				return tvpName;
			}

			if (!_token.TryIgnoreCase(SqlTokenizer.DataTypes, out var dataType))
			{
				throw new PrecursorException("<SqlDataType>");
			}

			var dataSize = Get(ParseDataTypeSize);

			TryGet(ParsePrimaryKey, out var primaryKeyExpr);

			return new DataTypeExpr
			{
				DataType = dataType,
				DataSize = dataSize,
				PrimaryKey = primaryKeyExpr
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
			ReadKeyword(")");

			return new TableTypeExpr
			{
				ColumnTypeList = columnTypeList,
			};
		}

		protected MaxExpr ParseMax()
		{
			if (!TryKeyword("MAX", out _))
			{
				throw new PrecursorException("MAX");
			}
			return new MaxExpr();
		}

		protected DataTypeSizeExpr ParseDataTypeSize()
		{
			if (!_token.Try("("))
			{
				throw new PrecursorException("(");
			}

			if (!TryGet(ParseMax, out SqlExpr size))
			{
				size = ParseInteger();
			}

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

		protected SqlExpr ParseSet_DEADLOCK_PRIORITY()
		{
			if (!TryAllKeywords(new[] { "SET", "DEADLOCK_PRIORITY" }, out _))
			{
				throw new PrecursorException("SET DEADLOCK_PRIORITY");
			}
			var actionName = ReadToken();
			return new SetOptionsExpr
			{
				Options = new List<string> { "DEADLOCK_PRIORITY" },
				Toggle = actionName
			};
		}

		protected string ReadToken()
		{
			var token = _token.Text;
			_token.Move();
			return token;
		}

		protected SetTransactionIsolationLevelExpr ParseSetTransaction()
		{
			if (!TryAllKeywords(new[] { "SET", "TRANSACTION", "ISOLATION", "LEVEL" }, out var tokens))
			{
				throw new PrecursorException("SET TRANSACTION");
			}

			if (!TryAllKeywords("READ UNCOMMITTED", out var actionNameExpr))
			{
				throw new ParseException();
			}

			var actionName = string.Join(" ", actionNameExpr);

			return new SetTransactionIsolationLevelExpr
			{
				ActionName = actionName
			};
		}

		protected SqlExpr ParseSet()
		{
			return Any("<SET xxx>",
				ParseSetTransaction,
				ParseSet_DEADLOCK_PRIORITY,
				ParseSetVariableEqual,
				ParseSet_Permission_ObjectId_OnOff,
				ParseSet_Options_OnOff,
				ParseSetvar
			);
		}

		protected InvokeFunctionExpr ParseFunctionWithParenthesesExpr(Func<SqlExpr> innerParse)
		{
			var startIndex = _token.CurrentIndex;
			if (!TryGet(() => (IdentExpr)Any("", ParseSqlIdent, ParseFuncName), out var funcName))
			{
				throw new PrecursorException("<FUNCTION NAME>");
			}

			if (!TryKeyword("(", out _))
			{
				_token.MoveTo(startIndex);
				throw new PrecursorException("(");
			}

			var argsList = WithComma(innerParse);

			ReadKeyword(")");
			return new InvokeFunctionExpr
			{
				Name = funcName,
				ArgumentsList = argsList
			};
		}


		protected InvokeFunctionExpr ParseFunctionWithParentheses()
		{
			return ParseFunctionWithParenthesesExpr(ParseNestExpr);
		}

		protected SetVariableExpr ParseSetVariableEqual()
		{
			var startIndex = _token.CurrentIndex;
			if (!TryKeyword("SET", out _))
			{
				throw new PrecursorException("SET");
			}

			if (!TryGet(ParseVariableName, out var variableName))
			{
				_token.MoveTo(startIndex);
				throw new PrecursorException("<VariableName>");
			}

			ReadKeyword("=");
			var valueExpr = Any("", ParseFunctionWithParentheses, ParseArithmeticExpr);

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

		public CreateSpExpr ParseCreateSpPartial(string sql)
		{
			return ParsePartial(ParseCreateSp, sql);
		}

		protected CreateSpExpr CreateSpSingleBody()
		{
			var startIndex = _token.CurrentIndex;
			if (!TryKeyword("CREATE", out _))
			{
				throw new PrecursorException("CREATE");
			}

			if (!TryKeyword("PROCEDURE", out _))
			{
				_token.MoveTo(startIndex);
				throw new PrecursorException("PROCEDURE");
			}

			var spName = ParseSqlIdent();
			var spArgs = ParseArgumentsList();

			ReadKeyword("AS");

			if (IsKeyword("BEGIN"))
			{
				_token.MoveTo(startIndex);
				throw new PrecursorException("<NO NEED BEGIN>");
			}

			var body = new[] { ParseExpr() }.ToList();
			return new CreateSpExpr
			{
				Name = spName,
				Arguments = spArgs,
				Body = body
			};
		}

		protected CreateSpExpr ParseCreateSp()
		{
			if (TryGet(CreateSpSingleBody, out var createSpSingleBodyExpr))
			{
				return createSpSingleBodyExpr;
			}

			var startIndex = _token.CurrentIndex;
			if (!TryKeyword("CREATE", out _))
			{
				throw new PrecursorException("CREATE");
			}

			if (!TryKeyword("PROCEDURE", out _))
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
						throw new Exception($"argument, '{_token.Text}'");
					}
					break;
				}

				TryKeyword("AS", out _);
				var dataType = ParseDataType();

				TryKeyword("OUTPUT", out var outputToken);

				SqlExpr defaultValue = null;
				if (_token.Try("="))
				{
					defaultValue = ParseConstant();
				}

				spArgs.Add(new ArgumentExpr
				{
					Name = argName,
					DataType = dataType,
					OutputToken = outputToken,
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

		public SqlExpr ParseInsertPartial(string sql)
		{
			return ParsePartial(ParseInsert, sql);
		}

		private SqlExprList EatInsertFields()
		{
			if (!_token.Try("("))
			{
				throw new PrecursorException($"'(' , but got {_token.Text}");
			}
			var fields = WithComma(ParseSqlIdent1);
			ReadKeyword(")");
			return fields;
		}

		protected MergeInsertExpr ParseMergeInsert()
		{
			var startIndex = _token.CurrentIndex;
			if (!TryKeyword("INSERT", out _))
			{
				throw new PrecursorException("INSERT");
			}
			if (!TryGet(() => EatInsertFields(), out var fields))
			{
				_token.MoveTo(startIndex);
				throw new PrecursorException("(<FIELDS>)");
			}
			var valuesList = ParseValuesList();
			return new MergeInsertExpr
			{
				Fields = fields,
				Values = valuesList
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

			TryGet(() => EatInsertFields(), out var fields);

			TryGet(ParseOutput, out var outputExpr);
			TryGet(ParseInto, out var inputExpr);

			if (IsKeyword("SELECT"))
			{
				var selectExpr = ParseSelect();
				return new InsertFromSelectExpr
				{
					IntoToggle = intoToggle,
					Table = table,
					OutputExpr = outputExpr,
					IntoExpr = inputExpr,
					FromSelect = selectExpr
				};
			}

			var valuesList = ParseValuesList();

			return new InsertExpr
			{
				IntoToggle = intoToggle,
				Table = table,
				Fields = fields,
				ValuesList = valuesList
			};
		}

		protected SqlExprList ParseValuesList()
		{
			ReadKeyword("VALUES");
			var valuesList = new SqlExprList()
			{
				Items = new List<SqlExpr>()
			};
			do
			{
				ReadKeyword("(");
				var values = WithComma(() => ParseParenthesesExpr());
				valuesList.Items.Add(values);
				ReadKeyword(")");
				if (!_token.Try(","))
				{
					break;
				}
			} while (true);
			return valuesList;
		}

		protected AssignSetExpr ParseFieldAssignValue()
		{
			var startIndex = _token.CurrentIndex;
			if (!TryGet(ParseSqlIdent, out var fieldExpr))
			{
				_token.MoveTo(startIndex);
				throw new PrecursorException("<Field>");
			}
			ReadKeyword("=");

			if (TryGet(ParseArithmeticExpr, out var arithemeticExpr))
			{
				return new AssignSetExpr
				{
					Field = fieldExpr,
					Value = arithemeticExpr
				};
			}

			if (TryGet(ParseSubExpr, out var subExpr))
			{
				return new AssignSetExpr
				{
					Field = fieldExpr,
					Value = subExpr
				};
			}

			throw new NotSupportedException("Field = xxx Expr");
		}

		public UpdateExpr ParseUpdatePartial(string sql)
		{
			return ParsePartial(ParseUpdate, sql);
		}

		protected SqlExpr ParseBetweenExpr(SqlExpr leftExpr)
		{
			if (!TryGet(ParseBetween, out var betweenExpr))
			{
				return leftExpr;
			}
			betweenExpr.Left = leftExpr;
			return betweenExpr;
		}

		protected BetweenExpr ParseBetween()
		{
			if (!TryKeyword("BETWEEN", out _))
			{
				throw new PrecursorException("BETWEEN");
			}
			var fromValue = ParseSubExpr();
			ReadKeyword("AND");
			var toValue = ParseSubExpr();
			return new BetweenExpr
			{
				From = fromValue,
				To = toValue,
			};
		}

		protected WaitforDelayExpr ParseWaitforDelay()
		{
			if (!TryAllKeywords(new[] { "WAITFOR", "DELAY" }, out _))
			{
				throw new PrecursorException("WAITFOR DELAY");
			}
			var value = ParseSubExpr();
			return new WaitforDelayExpr
			{
				Value = value,
			};
		}

		protected UpdateExpr ParseUpdate()
		{
			if (!_token.TryIgnoreCase("UPDATE"))
			{
				throw new PrecursorException("UPDATE");
			}

			TryGet(ParseTop, out var topExpr);

			var table = ParseSqlIdent();

			TryGet(ParseWithOptions, out var withOptions);

			ReadKeyword("SET");
			var setFields = WithComma(() => Any("<CASE> or <assign>", ParseCase, ParseFieldAssignValue));

			TryGet(ParseOutput, out var outputExpr);

			TryGet(ParseFrom, out var fromListExpr);

			var joinTableList = ParseInnerJonList();

			TryGet(ParseWhere, out var whereExpr);

			return new UpdateExpr
			{
				Top = topExpr,
				Table = table,
				WithOptions = withOptions,
				Fields = setFields,
				OutputExpr = outputExpr,
				FromTableList = fromListExpr,
				JoinTableList = joinTableList,
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

		protected DistinctExpr ParseDistinct()
		{
			if (!TryKeyword("DISTINCT", out _))
			{
				throw new PrecursorException("DISTINCT");
			}
			var rightSide = ParseArithmeticExpr();
			return new DistinctExpr
			{
				RightSide = rightSide,
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

			if (TryKeyword("UNION", out _))
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

		protected TopExpr ParseTop()
		{
			if (!TryKeyword("TOP", out _))
			{
				throw new PrecursorException("TOP");
			}

			var hasParentheses = false;
			if (TryKeyword("(", out _))
			{
				hasParentheses = true;
			}
			var count = ParseConstant();
			if (hasParentheses)
			{
				ReadKeyword(")");
			}

			return new TopExpr
			{
				HasParentheses = hasParentheses,
				Count = count
			};
		}

		protected OrderColumnExpr ParseColumnDesc()
		{
			//if (!TryGet(ParseSqlIdent, out var column))
			//{
			//	throw new PrecursorException("<Column>");
			//}

			if (!TryGet(ParseArithmeticExpr, out var column))
			{
				throw new PrecursorException("<Column>");
			}

			TryAnyKeywords(new[] { "ASC", "DESC" }, out var orderTypeExpr);

			return new OrderColumnExpr
			{
				Column = column,
				OrderType = orderTypeExpr
			};
		}

		protected SqlExprList ParseOrderBy()
		{
			if (!TryGet(Keywords("ORDER", "BY"), out _))
			{
				throw new PrecursorException("ORDER BY");
			}

			var columnList = WithComma(ParseColumnDesc);
			if (columnList.Items.Count <= 0)
			{
				throw new ParseException("<Column>");
			}

			return columnList;
		}

		protected SqlExprList GetMany(Func<SqlExpr> parse, bool hasComma = true)
		{
			if (!TryGet(() => Many(parse), out var exprList))
			{
				exprList = new SqlExprList()
				{
					HasComma = hasComma,
					Items = new List<SqlExpr>()
				};
			}
			return exprList;
		}

		protected SqlExprList ParseInnerJonList()
		{
			if (!TryGet(() => Many(ParseInnerJoin), out var joinTableList))
			{
				joinTableList = new SqlExprList()
				{
					Items = new List<SqlExpr>()
				};
			}
			return joinTableList;
		}

		protected IntoNewTableExpr ParseInto_NewTable()
		{
			if(!TryKeyword("INTO", out _))
			{
				throw new PrecursorException();
			}

			var newTable = ParseSqlIdent();

			return new IntoNewTableExpr
			{
				Table = newTable,
			};
		}

		protected SelectExpr ParseSelect()
		{
			if (!_token.TryIgnoreCase("select", out var _))
			{
				throw new PrecursorException();
			}

			TryGet(ParseTop, out var topExpr);

			var fields = ParseManyColumns();

			TryGet(ParseInto_NewTable, out var intoNewTableExpr);

			TryGet(ParseFrom, out var fromExpr);

			//
			var fromJoinTableList = WithComma(() => Any("<INNER JOIN> or <FROM JOIN>", ParseInnerJoin, ParseFromJoin));

			var innerJoinTableList = ParseInnerJonList();

			var whereExpr = Get(ParseWhere);

			TryGet(ParseGroupBy, out var groupByExpr);

			TryGet(ParseOrderBy, out var orderByExpr);

			var joinAllList = Many(ParseUnionJoinAll);

			return new SelectExpr
			{
				TopExpr = topExpr,
				Fields = fields,
				From = fromExpr,
				FromJoinList = fromJoinTableList,
				Joins = innerJoinTableList.Items,
				WhereExpr = whereExpr,
				GroupByExpr = groupByExpr,
				OrderByExpr = orderByExpr,
				IntoNewTable = intoNewTableExpr,
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
				TryGet(ParseAlias, out var aliasExpr);
				TryGet(ParseWithOptions, out var withOptionsExpr);

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
			if (!TryGet(ParseSqlIdent, out var tableName))
			{
				throw new PrecursorException("<TableName>");
			}

			TryGet(GetAliasName, out var aliasName);
			TryGet(ParseWithOptions, out var withOptions);
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
				if (!TryGet(parse, out var itemExpr))
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

		private List<T> Many<T>(Func<T> parse, int maxCount)
		{
			var list = new List<T>();
			do
			{
				if (!TryGet(parse, out var itemExpr))
				{
					break;
				}
				list.Add(itemExpr);
				if (list.Count >= maxCount)
				{
					break;
				}
			} while (true);
			return list;
		}

		private List<T> ManyWithComma<T>(Func<T> parse, int maxCount)
		{
			var list = new List<T>();
			do
			{
				if (!TryGet(parse, out var itemExpr))
				{
					break;
				}
				list.Add(itemExpr);
				if (list.Count >= maxCount)
				{
					break;
				}
				if (!_token.Try(","))
				{
					break;
				}
			} while (true);
			return list;
		}

		private SqlExprList WithComma<T>(Func<T> parse)
		where T : SqlExpr
		{
			var list = new List<SqlExpr>();
			do
			{
				var item = default(T);

				try
				{
					item = parse();
				}
				catch (PrecursorException)
				{
					break;
				}

				if (item is SqlExprList itemList)
				{
					//return itemList;
					if (itemList.Items.Count > 0)
					{
						list.Add(item);
					}
				}
				else
				{
					list.Add(item);
				}

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
			if (TryGet(ParseArithmeticExpr, out var arithmeticExpr))
			{
				return ParseAliasExpr(ParseSimpleColumnExpr(ParseEqualExpr(arithmeticExpr)));
			}

			if (TryGet(ParseConstant, out var constantExpr))
			{
				return ParseAliasExpr(ParseSimpleColumnExpr(ParseEqualExpr(constantExpr)));
			}

			if (_token.IgnoreCase("NOT"))
			{
				return ParseNot();
			}

			if (TryGet(ParseVariableName, out var variableName))
			{
				ReadKeyword("=");
				return new ColumnSetExpr
				{
					SetVariableName = variableName,
					Column = Any("<SimpleColumn> or <constant>",
						ParseArithmeticExpr,
						ParseSimpleColumn,
						ParseConstant, ParseSubExpr),
				};
			}

			if (TryGet(ParseArithmeticExpr, out var subExprColumn))
			{
				return ParseSimpleColumnExpr(subExprColumn);
			}

			return ParseSimpleColumn();
		}

		protected BreakExpr ParseBreak()
		{
			if (!TryKeyword(";", out _))
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
			WithAsItemExpr parseAliasName()
			{
				var startIndex = _token.CurrentIndex;
				if (!TryGet(ParseIdent, out var cteTableName))
				{
					throw new PrecursorException("<CTE TableName>");
				}

				var cteColumns = new SqlExprList();
				if (TryKeyword("(", out _))
				{
					cteColumns = WithComma(ParseSqlIdent1);
					ReadKeyword(")");
				}

				TryGet(ParseSqlIdent1, out var aliasName);

				if (!TryKeyword("AS", out _))
				{
					throw new ParseException("AS");
				}

				ReadKeyword("(");
				var innerSide = ParseArithmeticExpr();
				ReadKeyword(")");
				return new WithAsItemExpr
				{
					TableName = cteTableName,
					Columns = cteColumns,
					AliasName = aliasName,
					InnerSide = innerSide,
				};
			}


			var startIndex = _token.CurrentIndex;
			if (!TryKeyword("WITH", out _))
			{
				throw new PrecursorException("WITH");
			}

			//var innerExpr = ParseArithmeticExpr();
			var innerExpr = WithComma(parseAliasName);

			return new CommonTableExpressionExpr
			{
				InnerExpr = innerExpr,
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
			if (!TryGet(ParseAlias, out var aliasName))
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
			if (TryGet(ParseString, out var stringName))
			{
				return new IdentExpr
				{
					Name = stringName.Text
				};
			}

			if (TryGet(ParseSqlIdent1, out var aliasName))
			{
				return aliasName;
			}
			if (TryKeyword("AS", out _))
			{
				if (TryGet(ParseString, out var stringName2))
				{
					return new IdentExpr
					{
						Name = stringName2.Text
					};
				}
				return ParseIdent();
			}
			throw new PrecursorException("AS <Ident>");
		}

		private string GetAliasName()
		{
			var aliasNameExpr = ParseAlias();
			return aliasNameExpr.Name;

			//var aliasName = (string)null;
			//if (_token.IsIdent)
			//{
			//	aliasName = ParseIdent().Name;
			//}
			//if (_token.TryIgnoreCase("as"))
			//{
			//	aliasName = ParseIdent().Name;
			//}
			//return aliasName;
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

		public SqlExpr ParseSqlIdentPartial(string sql)
		{
			return ParsePartial(ParseSqlIdent, sql);
		}

		private string Parse_SqlIdent_Token()
		{
			if (!_token.Try(_token.IsSqlIdent, out var token))
			{
				throw new PrecursorException("<SqlIdent>");
			}
			return token;
		}

		private string Parse_Dot_SqlIdent_Token()
		{
			var startIndex = _token.CurrentIndex;
			if (!TryKeyword(".", out _))
			{
				throw new PrecursorException(".");
			}

			if (!TryGet(Parse_SqlIdent_Token, out var identToken))
			{
				_token.MoveTo(startIndex);
				throw new PrecursorException("<SqlIdent>");
			}

			return identToken;
		}

		protected IdentExpr ParseFuncName()
		{
			if (!_token.Try(_token.IsFuncName(out var funcArgsCount), out var funcName))
			{
				throw new PrecursorException("FUNCTION NAME");
			}
			return new IdentExpr
			{
				Name = funcName,
			};
		}

		protected IdentExpr ParseSqlIdent()
		{
			if (!_token.Try(_token.IsSqlIdent, out var s1))
			{
				throw new PrecursorException($"{_token.Text} should be <Ident>");
			}

			var dot_token_list = Many(Parse_Dot_SqlIdent_Token, 3);
			if (dot_token_list.Count == 0)
			{
				return new IdentExpr
				{
					Name = s1,
				};
			}

			switch (dot_token_list.Count)
			{
				case 1:
					return new IdentExpr
					{
						ObjectId = s1,
						Name = dot_token_list[0],
					};
				case 2:
					return new IdentExpr
					{
						DatabaseId = s1,
						ObjectId = dot_token_list[0],
						Name = dot_token_list[1],
					};
				case 3:
					return new IdentExpr
					{
						ServerId = s1,
						DatabaseId = dot_token_list[0],
						ObjectId = dot_token_list[1],
						Name = dot_token_list[2],
					};
			}

			throw new ParseException();
		}

		private ColumnExpr ParseSimpleColumnExpr(SqlExpr fieldExpr)
		{
			TryGet(ParseAlias, out var aliasName);

			return new ColumnExpr
			{
				Name = fieldExpr,
				AliasName = aliasName?.Name,
			};
		}

		protected RankOverExpr ParseRankOver()
		{
			if (!TryAllKeywords(new[] { "RANK", "(", ")", "OVER" }, out _))
			{
				throw new PrecursorException("RANK() OVER");
			}
			ReadKeyword("(");

			SqlExpr partitionBy = null;
			var partitionDescending = "";
			if (TryAllKeywords(new[] { "PARTITION", "BY" }, out var _))
			{
				partitionBy = ParseSubExpr();
				partitionDescending = ParseAscOrDescToken();
			}

			if (!TryAllKeywords(new[] { "ORDER", "BY" }, out _))
			{
				throw new ParseException("ORDER BY");
			}
			var orderByList = WithComma(ParseColumnDescending);
			ReadKeyword(")");

			var aliasName = ParseAlias();

			return new RankOverExpr
			{
				PartitionBy = partitionBy,
				PartitionDescending = partitionDescending,
				OrderByList = orderByList,
				AliasName = aliasName
			};
		}

		protected ColumnDescendingExpr ParseColumnDescending()
		{
			if (!TryGet(ParseSqlIdent, out var ident))
			{
				throw new PrecursorException("<ObjectId>");
			}
			var descending = ParseAscOrDescToken();
			return new ColumnDescendingExpr
			{
				Name = ident,
				Descending = descending,
			};
		}

		protected string ParseAscOrDescToken()
		{
			if (TryKeyword("ASC", out _))
			{
				return "ASC";
			}
			if (TryKeyword("DESC", out _))
			{
				return "DESC";
			}
			return string.Empty;
		}

		private ColumnExpr ParseSimpleColumn()
		{
			if (!TryGet(ParseSqlIdent, out var identExpr))
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

		protected OutputColumnExpr Parse_OutputOther()
		{
			if (!TryGet(ParseSubExpr, out var outputExpr))
			{
				throw new PrecursorException("<OUTPUT Column EXPR>");
			}

			TryGet(ParseAlias, out var aliasName);

			return new OutputColumnExpr
			{
				ActionName = String.Empty,
				Column = outputExpr,
				Alias = aliasName
			};
		}

		protected OutputColumnExpr Parse_OutputInsert()
		{
			if (!TryAnyKeywords(new[] { "INSERTED", "DELETED" }, out var token))
			{
				throw new PrecursorException("INSERTED or DELETED");
			}

			ReadKeyword(".");

			var columnName = ParseSqlIdent();
			TryGet(ParseAlias, out var aliasName);

			return new OutputColumnExpr
			{
				ActionName = token,
				Column = columnName,
				Alias = aliasName,
			};
		}

		protected OutputExpr ParseOutput()
		{
			if (!TryKeyword("OUTPUT", out _))
			{
				throw new PrecursorException("OUTPUT");
			}

			var outputList = WithComma(() => Any("<OUTPUT COLUMNS>", Parse_OutputInsert, Parse_OutputOther));
			if (outputList.Items.Count == 0)
			{
				throw new ParseException("<INSERTED EXPR>");
			}

			return new OutputExpr
			{
				ColumnList = outputList,
			};
		}


		protected IntoExpr ParseInto()
		{
			if (!TryKeyword("INTO", out _))
			{
				throw new PrecursorException("INTO");
			}

			var table = ParseSqlIdent();
			ReadKeyword("(");
			var columns = WithComma(ParseSqlIdent1);
			ReadKeyword(")");

			return new IntoExpr
			{
				Table = table,
				Columns = columns,
			};
		}

		protected WithOptionsExpr ParseWithOptions()
		{
			if (!_token.TryIgnoreCase("with"))
			{
				throw new PrecursorException("WITH");
			}

			if (!_token.Try("("))
			{
				throw new Exception("(");
			}

			string eatWithOption()
			{
				if (!TryAnyKeywords(new[] { "NOLOCK", "ROWLOCK", "UPDLOCK" }, out var token))
				{
					throw new PrecursorException("NOLOCK or ROWLOCK");
				}
				return token;
			}
			var withOptions = ManyWithComma(eatWithOption, 10);

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

		protected MarkPrimaryKeyExpr ParsePrimaryKey()
		{
			if (!TryAllKeywords(new[] { "PRIMARY", "KEY" }, out _))
			{
				throw new PrecursorException("PRIMARY KEY");
			}
			return new MarkPrimaryKeyExpr();
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

		protected BeginTransactionExpr ParseBegin_Transaction()
		{
			if (!TryAllKeywords(new[] { "BEGIN", "TRANSACTION" }, out _))
			{
				throw new PrecursorException("BEGIN TRANSACTION");
			}
			return new BeginTransactionExpr();
		}

		protected SqlExpr ParseBegin()
		{
			if (TryGet(ParseBegin_Transaction, out var transactionExpr))
			{
				return transactionExpr;
			}

			if (!TryKeyword("BEGIN", out _))
			{
				throw new PrecursorException("BEGIN");
			}
			var body = ParseBody();
			ReadKeyword("END");
			return new BeginExpr
			{
				Body = body
			};
		}

		protected SqlExpr ParseBeginEndOrOneExpr()
		{
			if (!TryGet(ParseBegin, out SqlExpr body))
			{
				body = ParseExpr();
			}
			return body;
		}

		protected ElseIfExpr ParseElseIf()
		{
			if (!TryAllKeywords(new[] { "ELSE", "IF" }, out _))
			{
				throw new PrecursorException("ELSE IF");
			}

			var condition = ParseFilterList();
			var body = ParseBeginEndOrOneExpr();

			return new ElseIfExpr
			{
				Condition = condition,
				Body = body
			};
		}

		protected SqlExpr ParseIf()
		{
			if (!TryKeyword("IF", out _))
			{
				throw new PrecursorException("IF");
			}
			var condition = ParseFilterList();

			var body = ParseBeginEndOrOneExpr();

			var elseIfList = Many(ParseElseIf);

			SqlExpr elseBody = null;
			if (TryKeyword("ELSE", out _))
			{
				elseBody = ParseBeginEndOrOneExpr();
			}

			return new IfExpr
			{
				Condition = condition,
				Body = body,
				ElseIfList = elseIfList,
				ElseBody = elseBody
			};
		}

		private List<SqlExpr> ParseBody()
		{
			var body = new List<SqlExpr>();
			do
			{
				if (IsKeyword("END"))
				{
					break;
				}
				var expr = Get(ParseExpr);
				//TryGet(ParseExpr, out var expr);
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
				//var filter = ParseParenthesesExpr();
				var filter = ParseNestExpr();
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

		private SqlExpr ParseConstantForReturn()
		{
			return Any(
				"",
				ParseNull,
				ParseNegativeNumber,
				ParseHex16Number,
				ParseDecimal,
				ParseInteger,
				ParseString,
				ParseVariable,
				ParseStar);
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
			return ParsePartial(ParseFilter, sql);
		}

		private SqlExpr ParseWithCompareOp(Func<SqlExpr> leftParse)
		{
			var leftExpr = leftParse();

			var likeExpr = Get(ParseLike, leftExpr);
			if (likeExpr != null)
			{
				return likeExpr;
			}

			leftExpr = ParseBetweenExpr(leftExpr);

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
			//if (IsKeyword("("))
			//{
			//	return ParseParentheses();
			//}

			if (IsKeyword("NOT"))
			{
				return ParseNot();
			}

			var left = ParseArithmeticExpr();
			return left;
		}

		public ExecuteExpr ParseExecPartial(string sql)
		{
			return ParsePartial(ParseExec, sql);
		}

		protected ExecuteExpr Parse_variable_eq_function()
		{
			if (!_token.TryMatch(SqlTokenizer.SqlVariable, out var variableName))
			{
				throw new PrecursorException();
			}
			ReadKeyword("=");
			var execExpr = Eat_function_args();
			execExpr.LeftSide = variableName;
			return execExpr;
		}

		protected ExecuteExpr ParseExec()
		{
			if (!_token.TryIgnoreCase(new[] { "EXECUTE", "EXEC" }, out var execStr))
			{
				throw new PrecursorException("Expect EXEC");
			}

			if(TryKeyword("(", out _))
			{
				var varName = ParseVariable();
				ReadKeyword(")");
				return new ExecuteExpr
				{
					ExecName = execStr,
					Method = new GroupExpr
					{
						InnerExpr = varName 
					},
				};
			}

			if (TryGet(Parse_variable_eq_function, out var variableEqExpr))
			{
				variableEqExpr.ExecName = execStr;
				return variableEqExpr;
			}

			var execExpr = Eat_function_args();
			execExpr.ExecName = execStr;

			return execExpr;
		}

		protected ExecuteExpr Eat_function_args()
		{
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

		private bool TryGet<T>(Func<T> parse, out T output)
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

		private bool TryGetAny(out SqlExpr output, params Func<SqlExpr>[] parseList)
		{
			for (var i = 0; i < parseList.Length; i++)
			{
				if (TryGet(parseList[i], out output))
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

		private SqlExpr ParseCompareOp(SqlExpr left, Func<SqlExpr> parseRight)
		{
			var op = string.Empty;
			SqlExpr parseAction()
			{
				SqlExpr right = null;
				if (op.IsSql("in"))
				{
					right = ParseParentheses();
				}
				else
				{
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

		public SqlExpr ParseParenthesesPartial(string sql)
		{
			return ParsePartial(ParseParentheses, sql);
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
				InnerExpr = innerExpr,
			};
		}

		protected SqlExpr ParseNestExpr()
		{
			return ParseWithCompareOp(ParseArithmeticExpr);
		}

		protected SqlExpr ParseParenthesesExpr(Func<SqlExpr> innerParse)
		{
			if (!TryKeyword("(", out _))
			{
				throw new PrecursorException("(");
			}

			var innerExpr = innerParse();

			ReadKeyword(")");

			var groupExpr = new GroupExpr
			{
				InnerExpr = innerExpr,
			};
			return ParseCompareOpExpr(groupExpr);
		}

		protected SqlExpr ParseParentheses()
		{
			return ParseArithmeticExpr();
			//return ParseParenthesesExpr(ParseNestExpr);
			//if (!TryKeyword("(", out _))
			//{
			//	throw new PrecursorException("(");
			//}
			//var subExpr = ParseNestExpr();

			//ReadKeyword(")");

			//var groupExpr = new GroupExpr
			//{
			//	Expr = subExpr,
			//};
			//return ParseCompareOpExpr(groupExpr);
		}

		protected SqlExpr ParseParenthesesExpr()
		{
			if (!TryGet(ParseParentheses, out var parenthesesExpr))
			{
				return ParseNestExpr();
			}
			return parenthesesExpr;
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


		private FromJoinExpr ParseFromJoin()
		{
			if (!TryGet(ParseSqlIdent, out var fromTable))
			{
				throw new PrecursorException("<TABLE>");
			}
			TryGet(ParseAlias, out var aliasName);
			TryGet(ParseWithOptions, out var withOptions);

			return new FromJoinExpr
			{
				Table = fromTable,
				AliasName = aliasName,
				WithOptions = withOptions
			};
		}

		private InnerJoinExpr ParseInnerJoin()
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
				_token.MoveTo(startIndex);
				throw new PrecursorException("JOIN");
			}

			ReadKeyword("JOIN");

			if (IsKeyword("ALL"))
			{
				_token.MoveTo(startIndex);
				throw new PrecursorException("<TABLE>");
			}

			//var table = ParseTableToken();
			var table = Any("<SubExpr> or <Table>", ParseParentheses, ParseTableToken);

			TryGet(ParseAlias, out var aliasName);
			TryGet(ParseWithOptions, out var withOptions);

			ReadKeyword("ON");

			var filterList = ParseFilterList();

			var joinType = JoinType.Inner;
			if (!string.IsNullOrEmpty(joinTypeStr))
			{
				joinType = (JoinType)Enum.Parse(typeof(JoinType), joinTypeStr, true);
			}

			return new InnerJoinExpr
			{
				Table = table,
				AliasName = aliasName,
				WithOptions = withOptions,
				JoinType = joinType,
				OuterToken = outerToken,
				OnFilter = filterList
			};
		}

		protected CommitExpr ParseCommit()
		{
			if (!TryKeyword("COMMIT", out _))
			{
				throw new PrecursorException("COMMIT");
			}
			return new CommitExpr();
		}

		public SqlExpr ParseParameterNameAssign()
		{
			if (!_token.TryMatch(SqlTokenizer.SqlVariable, out var varName))
			{
				throw new PrecursorException("Expect SqlVariable");
			}

			var outToken = string.Empty;
			if (!_token.Try("="))
			{
				TryKeyword("OUT", out outToken);
				return new SpParameterExpr
				{
					Name = varName,
					OutToken = outToken?.ToUpper()
				};
			}

			var value = ParseConstant();
			TryKeyword("OUT", out outToken);
			return new SpParameterExpr
			{
				Name = varName,
				Value = value,
				OutToken = outToken?.ToUpper(),
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

		public SqlExpr ParseAlterPartial(string sql)
		{
			return ParsePartial(ParseAlter, sql);
		}

		protected SqlExpr ParseAlter()
		{
			if (!TryKeyword("ALTER", out _))
			{
				throw new PrecursorException("ALTER");
			}

			if (TryGet(EatDatabase, out var alterDatabaseExpr))
			{
				return alterDatabaseExpr;
			}

			throw new ParseException($"'{_token.Text}' Not Support");
		}

		protected AlterDatabaseExpr EatDatabase()
		{
			if (!TryKeyword("DATABASE", out _))
			{
				throw new PrecursorException("DATABASE");
			}

			var dbNameExpr = ParseSqlIdent1();
			if (!TryAllKeywords(new[] { "ADD", "FILEGROUP" }, out _))
			{
				throw new ParseException("ADD FILEGROUP");
			}

			var filegroupName = ParseSqlIdent1();
			return new AlterDatabaseExpr
			{
				DbName = dbNameExpr,
				ActionName = "ADD",
				FileGroupName = filegroupName,
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
