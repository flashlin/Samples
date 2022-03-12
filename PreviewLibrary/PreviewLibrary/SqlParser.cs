using PreviewLibrary.Exceptions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using T1.Standard.Extensions;

namespace PreviewLibrary
{
	public class SqlParser
	{
		private SqlTokenizer _token;
		private string _sql;

		public SqlParser()
		{
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

		public IEnumerable<SqlExpr> ParseAllExpr()
		{
			do
			{
				var expr = ParseExpr();
				yield return expr;
			} while (!string.IsNullOrEmpty(_token.Text));
		}

		public SqlExpr ParseExpr()
		{
			var parses = new Func<SqlExpr>[]
			{
				ParseSemicolon,
				ParseSelect,
				ParseInsert,
				ParseMultiLineComment,
				ParseGo,
				ParseSet_Permission_ObjectId_OnOff,
				ParseSet_Options_OnOff,
				ParseSetvar,
				ParseOnCondition,
				ParseIf,
				ParsePrint,
				ParseUse,
				ParseExec,
				ParseGrant
			};
			for (var i = 0; i < parses.Length; i++)
			{
				if (TryGet(parses[i], out var expr))
				{
					return expr;
				}
			}
			throw new NotSupportedException(GetLastLineCh());
		}

		protected GrantToExpr ParseGrant()
		{
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
			if (TryGet(ParseNot, out var notExpr))
			{
				return notExpr;
			}
			if (TryGet(ParseCast, out var castExpr))
			{
				return castExpr;
			}
			if (_token.IsFuncName(out _))
			{
				return ParseSqlFunc();
			}
			var constantExpr = Get(ParseConstant);
			if (constantExpr != null)
			{
				return constantExpr;
			}
			if (TryGet(ParseSelect, out var selectExpr))
			{
				return selectExpr;
			}
			if (TryGet(ParseExec, out var execExpr))
			{
				return execExpr;
			}
			throw new Exception(GetLastLineCh() + " Expect sub expr");
		}

		protected NotExpr ParseNot()
		{
			if (!_token.TryIgnoreCase("NOT"))
			{
				throw new PrecursorException("Expect NOT");
			}
			var right = ParseSubExpr();
			return new NotExpr
			{
				Right = right
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
				var expr = ParseSubExpr();
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

		protected DataTypeExpr ParseDataType()
		{
			var dataTypes = new string[]
			{
				"INT", "DATETIME"
			};
			if (!_token.TryIgnoreCase(dataTypes, out var dataType))
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
			sb.AppendLine($"Line:{lnch.LineNumber} Ch:{lnch.ChNumber} ErrorToken:'{_token.Text}'");
			sb.AppendLine();
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

		protected SetBatchVariableExpr ParseSetvar()
		{
			if (!_token.TryIgnoreCase(":setvar"))
			{
				throw new PrecursorException($"Expect ':setvar', but got '{_token.Text}'");
			}

			var name = ParseIdent().Name;
			var value = _token.Read(SqlTokenizer.SqlDoubleQuotedString, nameof(SqlTokenizer.SqlDoubleQuotedString));
			return new SetBatchVariableExpr
			{
				Name = name,
				Value = value
			};
		}

		protected InsertValuesExpr ParseInsert()
		{
			var startIndex = _token.CurrentIndex;
			if (!_token.TryIgnoreCase("INSERT"))
			{
				throw new PrecursorException("INSERT");
			}

			var table = Get(ParseSqlIdent);
			if (table == null)
			{
				_token.MoveTo(startIndex);
				throw new PrecursorException("<table>");
			}

			if (!_token.Try("("))
			{
				_token.MoveTo(startIndex);
				throw new PrecursorException("(");
			}
			var fields = WithComma(ParseSqlIdent1);
			ReadKeyword(")");


			ReadKeyword("VALUES");

			var valuesList = new List<List<SqlExpr>>();
			do
			{
				ReadKeyword("(");
				var values = WithComma(() => Any("Constant or FUNC", ParseSqlFunc, ParseConstant));
				valuesList.Add(values);
				ReadKeyword(")");
				if (!_token.Try(","))
				{
					break;
				}
			} while (true);

			return new InsertValuesExpr
			{
				Table = table,
				Fields = fields,
				ValuesList = valuesList
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
			options.AddRange(optionExprs.Select(x => x.Name));
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


		private MultiLineCommentExpr ParseMultiLineComment()
		{
			if (!_token.Try(_token.IsMultiLineComment, out var str))
			{
				throw new PrecursorException("should /* */");
			}
			return new MultiLineCommentExpr
			{
				Text = str,
			};
		}

		public SelectExpr ParseSelect()
		{
			if (!_token.TryIgnoreCase("select", out var _))
			{
				throw new PrecursorException();
			}

			var fields = ParseManyColumns();
			var fromExpr = Get(ParseFrom);
			var whereExpr = Get(ParseWhere);
			var joinTable = Get(ParseJoin);
			var joinTableList = (List<JoinExpr>)null;
			if (joinTable != null)
			{
				joinTableList = new List<JoinExpr>
				{
					joinTable,
				};
			}

			return new SelectExpr
			{
				Fields = fields,
				From = fromExpr,
				WhereExpr = whereExpr,
				Joins = joinTableList
			};
		}

		private SqlExpr ParseFrom()
		{
			if (!_token.TryIgnoreCase("from", out var _))
			{
				throw new Exception();
			}
			return ParseTableToken();
		}

		private TableExpr ParseTableToken()
		{
			var tableName = ParseSqlIdent();
			var aliasName = GetAliasName();
			var withOptions = Get(ParseWithOptions);
			return new TableExpr
			{
				Name = tableName,
				AliasName = aliasName,
				WithOptions = withOptions
			};
		}

		private List<T> WithComma<T>(Func<T> parse)
		{
			var list = new List<T>();
			do
			{
				var item = parse();
				list.Add(item);
				if (!_token.Try(","))
				{
					break;
				}
			} while (true);
			return list;
		}

		private List<SqlExpr> ParseManyColumns()
		{
			var columns = WithComma(ParseSelectColumn);
			if (columns.Count == 0)
			{
				throw new Exception("field");
			}
			return columns;
		}

		private SqlExpr ParseSelectColumn()
		{
			if (_token.IsNumber)
			{
				return ParseInteger();
			}
			if (_token.IgnoreCase("NOT"))
			{
				return ParseNot();
			}
			return ParseSimpleColumn();
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
				throw new Exception($"{_token.Text} should be <Ident>");
			}
			return new IdentExpr
			{
				Name = s1
			};
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

		private ColumnExpr ParseSimpleColumn()
		{
			var identExpr = ParseSqlIdent();
			var aliasName = GetAliasName();

			return new ColumnExpr
			{
				Database = identExpr.DatabaseId,
				Table = identExpr.ObjectId,
				Name = identExpr.Name,
				AliasName = aliasName,
			};
		}

		protected Hex16NumberExpr ParseHex16Number()
		{
			if(!_token.TryMatch(SqlTokenizer.Hex16Number, out var hex))
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
				throw new Exception("WITH");
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

		public SqlExpr ParseWhere()
		{
			if (!_token.TryIgnoreCase("where"))
			{
				throw new PrecursorException("should is 'WHERE'");
			}
			return ParseFilterList();
		}

		protected SqlExpr ParseIf()
		{
			if (!_token.TryIgnoreCase("IF", out var ifStr))
			{
				throw new PrecursorException("Expect IF");
			}
			var filter = ParseFilterList();
			var body = new List<SqlExpr>();
			ReadKeyword("BEGIN");
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
			return new IfExpr
			{
				Condition = filter,
				Body = body,
			};
		}

		private SqlExpr ParseFilterList()
		{
			var postfixExprs = new List<object>();
			var ops = new Stack<string>();
			do
			{
				var filter = ParseFilter();
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
					var expr = new AndOrExpr
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

		private SqlExpr ParseConstant()
		{
			var expr = GetAny(ParseNull, ParseHex16Number, ParseString, ParseInteger, ParseSqlIdent);
			if (expr == null)
			{
				ThrowLastLineCh("Expect constant");
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

		private SqlExpr ParseFilter()
		{
			if (_token.IgnoreCase("NOT"))
			{
				return ParseNot();
			}

			var left = ParseSubExpr();
			var likeExpr = Get(ParseLike, left);
			if (likeExpr != null)
			{
				return likeExpr;
			}

			var compareExpr = Get(ParseCompareOp, left);
			if (compareExpr != null)
			{
				return compareExpr;
			}

			throw new NotSupportedException(GetLastLineCh());
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

		private bool TryGet<T>(Func<T> parse, out T output)
		{
			output = Get<T>(parse);
			return output != null;
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

		private SqlExpr Any(string expect, params Func<SqlExpr>[] parseList)
		{
			var expr = GetAny(parseList);
			if (expr == null)
			{
				throw new Exception(expect);
			}
			return expr;
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

		private CompareExpr ParseCompareOp(SqlExpr left)
		{
			var op = string.Empty;
			CompareExpr parseAction()
			{
				var right = ParseSubExpr(); //ParseSqlIdent();
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
				Throw("should be compare oper");
			}
			return parseAction();
		}

		private bool IsKeyword(string keyword)
		{
			return _token.IgnoreCase(keyword);
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

		private JoinExpr ParseJoin()
		{
			ReadKeywordOption(new[] { "inner", "left", "right" }, out var joinTypeStr);
			ReadKeyword("join");

			var table = ParseTableToken();
			var joinType = (JoinType)Enum.Parse(typeof(JoinType), joinTypeStr, true);

			ReadKeyword("on");
			var filterList = ParseFilterList();

			return new JoinExpr
			{
				Table = table,
				JoinType = joinType,
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
