using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;
using T1.CodeDom.TSql.Parselets;

namespace T1.CodeDom.TSql
{
	public class TSqlParser : PrattParser
	{
		public TSqlParser(IScanner scanner) : base(scanner)
		{
			Register(SqlToken.Asterisk, new AsteriskParselet());
			Register(SqlToken.Begin, new BeginParselet());
			Register(SqlToken.Cast, new CastParselet());
			Register(SqlToken.Case, new CaseParselet());
			Register(SqlToken.Convert, new ConvertParselet());
			Register(SqlToken.Create, new CreateParselet());
			Register(SqlToken.Commit, new CommitParselet());
			Register(SqlToken.Cursor, new CursorParselet());
			Register(SqlToken.Continue, new ContinueParselet());
			Register(SqlToken.DoubleQuoteString, new DoubleQuoteStringParselet());
			Register(SqlToken.Drop, new DropParselet());
			Register(SqlToken.Delete, new DeleteParselet());
			Register(SqlToken.Declare, new DeclareParselet());
			Register(SqlToken.Distinct, new DistinctParselet());
			Register(SqlToken.Exists, new ExistsParselet());
			Register(SqlToken.Exec, new ExecParselet());
			Register(SqlToken.Execute, new ExecParselet());
			Register(SqlToken.Fetch, new FetchParselet());
			Register(SqlToken.Go, new GoParselet());
			Register(SqlToken.Grant, new GrantParselet());
			Register(SqlToken.HexNumber, new HexNumberParselet());
			Register(SqlToken.Identifier, new ObjectIdParselet());
			Register(SqlToken.Insert, new InsertParselet());
			Register(SqlToken.IsNull, new IsNullParselet());
			Register(SqlToken.LParen, new GroupParselet());
			Register(SqlToken.Left, new LeftOrRightFunctionParselet());
			Register(SqlToken.Object, new ObjectParselet());
			Register(SqlToken.Open, new OpenParselet());
			Register(SqlToken.Pivot, new PivotParselet());
			Register(SqlToken.Print, new PrintParselet());
			Register(SqlToken.QuoteString, new QuoteStringParselet());
			Register(SqlToken.Rank, new RankParselet());
			Register(SqlToken.ROW_NUMBER, new RowNumberParselet());
			Register(SqlToken.Right, new LeftOrRightFunctionParselet());
			Register(SqlToken.ROLLBACK, new RollbackParselet());
			Register(SqlToken.Select, new SelectParselet());
			Register(SqlToken.SqlIdentifier, new ObjectIdParselet());
			Register(SqlToken.Set, new SetParselet());
			Register(SqlToken.SystemVariable, new SystemVariableParselet());
			Register(SqlToken.SingleComment, new CommentParselet());
			Register(SqlToken.Semicolon, new SemicolonParselet());
			Register(SqlToken.ScriptSetVar, new ScriptSetvarParselet());
			Register(SqlToken.ScriptOn, new ScriptOnParselet());
			Register(SqlToken.Source, new SourceParselet());
			Register(SqlToken.Target, new TargetParselet());
			Register(SqlToken.TempTable, new TempTableParselet());
			Register(SqlToken.Truncate, new TruncateTableParselet());
			Register(SqlToken.Number, new NumberParselet());
			Register(SqlToken.NString, new NStringParselet());
			Register(SqlToken.Not, new NotParselet());
			Register(SqlToken.Null, new NullParselet());
			Register(SqlToken.Merge, new MergeParselet());
			//Register(SqlToken.MAX, new FuncParselet(1));
			Register(SqlToken.MultiComment, new CommentParselet());
			Register(SqlToken.Update, new UpdateParselet());
			Register(SqlToken.Variable, new VariableParselet());
			Register(SqlToken.With, new WithParselet());
			Register(SqlToken.While, new WhileParselet());

			Register(SqlToken.ABS, new CallFuncParselet(1, 1));
			Register(SqlToken.CHARINDEX, new CallFuncParselet(2, 3));
			Register(SqlToken.COUNT, new CallFuncParselet(1, 1));
			Register(SqlToken.COALESCE, new CallFuncParselet(1));
			Register(SqlToken.DATEADD, new DateAddFuncParselet());
			Register(SqlToken.DATEPART, new DatePartFuncParselet());
			Register(SqlToken.DATEDIFF, new DateDiffFuncParselet());
			Register(SqlToken.DAY, new CallFuncParselet(1, 1));
			Register(SqlToken.EXP, new CallFuncParselet(1, 1));
			Register(SqlToken.FLOOR, new CallFuncParselet(1, 1));
			Register(SqlToken.GETDATE, new CallFuncParselet());
			Register(SqlToken.LEN, new CallFuncParselet(1, 1));
			Register(SqlToken.LOG, new CallFuncParselet(1, 1));
			Register(SqlToken.LOWER, new CallFuncParselet(1, 1));
			Register(SqlToken.MIN, new CallFuncParselet(1, 1));
			Register(SqlToken.MAX, new CallFuncParselet(1));
			Register(SqlToken.MONTH, new CallFuncParselet(1));
			Register(SqlToken.ROUND, new CallFuncParselet(2, 3));
			Register(SqlToken.REPLACE, new CallFuncParselet(3, 3));
			Register(SqlToken.RAISERROR, new CallFuncParselet(1));
			Register(SqlToken.SUM, new CallFuncParselet(1, 1));
			Register(SqlToken.SUSER_SNAME, new CallFuncParselet(1, 1));
			Register(SqlToken.SUBSTRING, new CallFuncParselet(3, 3));
			Register(SqlToken.YEAR, new CallFuncParselet(1, 1));

			Register(SqlToken.Between, new BetweenParselet());
			Register(SqlToken.Not, new NotInfixParselet());
			Register(SqlToken.If, new IfParselet());
			Register(SqlToken.Like, new LikeParselet());
			Register(SqlToken.In, new InParselet());

			Prefix(SqlToken.Plus, Precedence.PREFIX);
			Prefix(SqlToken.Minus, Precedence.PREFIX);
			Prefix(SqlToken.Tilde, Precedence.PREFIX);

			InfixLeft(SqlToken.Equal, Precedence.COMPARE);
			InfixLeft(SqlToken.And, Precedence.CONCAT);
			InfixLeft(SqlToken.Or, Precedence.CONCAT);
			InfixLeft(SqlToken.Is, Precedence.COMPARE);
			InfixLeft(SqlToken.NotEqual, Precedence.COMPARE);
			InfixLeft(SqlToken.SmallerThan, Precedence.COMPARE);
			InfixLeft(SqlToken.BiggerThan, Precedence.COMPARE);
			InfixLeft(SqlToken.BiggerThanOrEqual, Precedence.COMPARE);
			InfixLeft(SqlToken.SmallerThanOrEqual, Precedence.COMPARE);

			InfixLeft(SqlToken.Plus, Precedence.SUM);
			InfixLeft(SqlToken.Minus, Precedence.SUM);
			InfixLeft(SqlToken.Asterisk, Precedence.PRODUCT);
			InfixLeft(SqlToken.Ampersand, Precedence.PRODUCT);
			InfixLeft(SqlToken.Slash, Precedence.PRODUCT);
			InfixLeft(SqlToken.VerticalBar, Precedence.BINARY);
			InfixLeft(SqlToken.Caret, Precedence.EXPONENT);
			InfixLeft(SqlToken.PlusEqual, Precedence.PREFIX);
			InfixLeft(SqlToken.MinusEqual, Precedence.PREFIX);
		}

		public SqlCodeExpr ParseExpression()
		{
			return (SqlCodeExpr)ParseExp(0);
		}

		protected void Register(SqlToken tokenType, IPrefixParselet parselet)
		{
			Register(tokenType.ToString(), parselet);
		}

		protected void Register(SqlToken tokenType, IInfixParselet parselet)
		{
			Register(tokenType.ToString(), parselet);
		}

		protected void Prefix(SqlToken tokenType, Precedence precedence)
		{
			Register(tokenType.ToString(), new SqlPrefixOperatorParselet(precedence));
		}

		public void InfixLeft(SqlToken tokenType, Precedence precedence)
		{
			Register(tokenType.ToString(), new BinaryOperatorParselet(precedence, false));
		}

		protected override (TextSpan infixToken, int consumeIndex) PeekToken()
		{
			var index = _scanner.GetOffset();
			var lastConsumeIndex = index;
			TextSpan lastInfixToken = TextSpan.Empty;
			do
			{
				_scanner.SetOffset(index);
				var (infixToken, consumeIndex) = base.PeekToken();
				if (infixToken.IsEmpty)
				{
					break;
				}
				if (infixToken.Type == SqlToken.MultiComment.ToString() || infixToken.Type == SqlToken.SingleComment.ToString())
				{
					index = consumeIndex;
					continue;
				}
				lastInfixToken = infixToken;
				lastConsumeIndex = consumeIndex;
				break;
			} while (true);

			return (lastInfixToken, lastConsumeIndex);
		}

		protected override IPrefixParselet CodeSpecPrefix(TextSpan token)
		{
			try
			{
				return base.CodeSpecPrefix(token);
			}
			catch (KeyNotFoundException)
			{
				var tokenStr = _scanner.GetSpanString(token);

				var helpMessage = _scanner.GetHelpMessage(token);
				throw new ParseException($"Not found SqlType.{token.Type} '{tokenStr}' in PrefixParselets map.\r\n{helpMessage}");
			}
		}
	}
}
