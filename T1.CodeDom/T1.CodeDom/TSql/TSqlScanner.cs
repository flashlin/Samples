using System;
using System.Linq;
using System.Text;
using T1.CodeDom.Core;

namespace T1.CodeDom.TSql
{
	public class TSqlScanner : StringScanner
	{
		char[] _magnetHeadChars = new char[]
		{
			'>', '<'
		};

		public TSqlScanner(string text)
			: base(text)
		{
			AddTokenMap("AS", SqlToken.As);
			AddTokenMap("ASC", SqlToken.Asc);
			AddTokenMap("AND", SqlToken.And);
			AddTokenMap("ALL", SqlToken.All);
			AddTokenMap("BREAK", SqlToken.Break);
			AddTokenMap("BEGIN", SqlToken.Begin);
			AddTokenMap("BIT", SqlToken.Bit);
			AddTokenMap("BY", SqlToken.By);
			AddTokenMap("BETWEEN", SqlToken.Between);
			AddTokenMap("BIGINT", SqlToken.Bigint);
			AddTokenMap("CONTINUE", SqlToken.Continue);
			AddTokenMap("CHAR", SqlToken.Char);
			AddTokenMap("CAST", SqlToken.Cast);
			AddTokenMap("CASE", SqlToken.Case);
			AddTokenMap("CURSOR", SqlToken.Cursor);
			AddTokenMap("CROSS", SqlToken.Cross);
			AddTokenMap("CREATE", SqlToken.Create);
			AddTokenMap("CONVERT", SqlToken.Convert);
			AddTokenMap("CLUSTERED", SqlToken.Clustered);
			AddTokenMap("DROP", SqlToken.Drop);
			AddTokenMap("DESC", SqlToken.Desc);
			AddTokenMap("DISTINCT", SqlToken.Distinct);
			AddTokenMap("DECLARE", SqlToken.Declare);
			AddTokenMap("DELETE", SqlToken.Delete);
			AddTokenMap("DEFAULT", SqlToken.Default);
			AddTokenMap("DATE", SqlToken.Date);
			AddTokenMap("DATETIME", SqlToken.DateTime);
			AddTokenMap("DATETIME2", SqlToken.DateTime2);
			AddTokenMap("DECIMAL", SqlToken.Decimal);
			AddTokenMap("DELETED", SqlToken.Deleted);
			AddTokenMap("ERROR", SqlToken.Error);
			AddTokenMap("EXIT", SqlToken.Exit);
			AddTokenMap("END", SqlToken.End);
			AddTokenMap("EXISTS", SqlToken.Exists);
			AddTokenMap("EXECUTE", SqlToken.Execute);
			AddTokenMap("EXEC", SqlToken.Exec);
			AddTokenMap("ELSE", SqlToken.Else);
			AddTokenMap("FOR", SqlToken.For);
			AddTokenMap("FROM", SqlToken.From);
			AddTokenMap("FLOAT", SqlToken.Float);
			AddTokenMap("FULL", SqlToken.Full);
			AddTokenMap("FETCH", SqlToken.Fetch);
			AddTokenMap("FUNCTION", SqlToken.Function);
			AddTokenMap("GRANT", SqlToken.Grant);
			AddTokenMap("GO", SqlToken.Go);
			AddTokenMap("GROUP", SqlToken.Group);
			AddTokenMap("IF", SqlToken.If);
			AddTokenMap("IS", SqlToken.Is);
			AddTokenMap("ISNULL", SqlToken.IsNull);
			AddTokenMap("IN", SqlToken.In);
			AddTokenMap("INT", SqlToken.Int);
			AddTokenMap("INSERT", SqlToken.Insert);
			AddTokenMap("INTO", SqlToken.Into);
			AddTokenMap("INDEX", SqlToken.Index);
			AddTokenMap("INNER", SqlToken.Inner);
			AddTokenMap("INSERTED", SqlToken.Inserted);
			AddTokenMap("JOIN", SqlToken.Join);
			AddTokenMap("KEY", SqlToken.Key);
			AddTokenMap("LIKE", SqlToken.Like);
			AddTokenMap("LEFT", SqlToken.Left);
			AddTokenMap("NVARCHAR", SqlToken.NVarchar);
			AddTokenMap("NEXT", SqlToken.Next);
			AddTokenMap("NOT", SqlToken.Not);
			AddTokenMap("NULL", SqlToken.Null);
			AddTokenMap("NUMERIC", SqlToken.Numeric);
			AddTokenMap("MERGE", SqlToken.Merge);
			AddTokenMap("MATCHED", SqlToken.Matched);
			AddTokenMap("ON", SqlToken.On);
			AddTokenMap("OBJECT", SqlToken.Object);
			AddTokenMap("OFF", SqlToken.Off);
			AddTokenMap("OUT", SqlToken.Out);
			AddTokenMap("OR", SqlToken.Or);
			AddTokenMap("OPEN", SqlToken.Open);
			AddTokenMap("ORDER", SqlToken.Order);
			AddTokenMap("OUTER", SqlToken.Outer);
			AddTokenMap("OVER", SqlToken.Over);
			AddTokenMap("OUTPUT", SqlToken.Output);
			AddTokenMap("OPTION", SqlToken.Option);
			AddTokenMap("PARTITION", SqlToken.Partition);
			AddTokenMap("PRIMARY", SqlToken.Primary);
			AddTokenMap("PIVOT", SqlToken.Pivot);
			AddTokenMap("PROCEDURE", SqlToken.Procedure);
			AddTokenMap("PROC", SqlToken.Procedure);
			AddTokenMap("ROW_NUMBER", SqlToken.ROW_NUMBER);
			AddTokenMap("PRINT", SqlToken.Print);
			AddTokenMap("RANK", SqlToken.Rank);
			AddTokenMap("RETURNS", SqlToken.Returns);
			AddTokenMap("RIGHT", SqlToken.Right);
			AddTokenMap("RANGE", SqlToken.Range);
			AddTokenMap("READONLY", SqlToken.ReadOnly);
			AddTokenMap("SET", SqlToken.Set);
			AddTokenMap("SOURCE", SqlToken.Source);
			AddTokenMap("SCHEME", SqlToken.Scheme);
			AddTokenMap("SELECT", SqlToken.Select);
			AddTokenMap("SMALLDATETIME", SqlToken.SmallDateTime);
			AddTokenMap("TRUNCATE", SqlToken.Truncate);
			AddTokenMap("TO", SqlToken.To);
			AddTokenMap("TARGET", SqlToken.Target);
			AddTokenMap("TOP", SqlToken.Top);
			AddTokenMap("TABLE", SqlToken.Table);
			AddTokenMap("THEN", SqlToken.Then);
			AddTokenMap("TINYINT", SqlToken.TinyInt);
			AddTokenMap("UPDATE", SqlToken.Update);
			AddTokenMap("USING", SqlToken.Using);
			AddTokenMap("UNION", SqlToken.Union);
			AddTokenMap("VALUES", SqlToken.Values);
			AddTokenMap("VARCHAR", SqlToken.Varchar);
			AddTokenMap("WHERE", SqlToken.Where);
			AddTokenMap("WITH", SqlToken.With);
			AddTokenMap("WHEN", SqlToken.When);
			AddTokenMap("WHILE", SqlToken.While);
			AddTokenMap(":SETVAR", SqlToken.ScriptSetVar);
			AddTokenMap(":ON", SqlToken.ScriptOn);
			
			AddTokenMap("ABS", SqlToken.ABS);
			AddTokenMap("CHARINDEX", SqlToken.CHARINDEX);
			AddTokenMap("COUNT", SqlToken.COUNT);
			AddTokenMap("COALESCE", SqlToken.COALESCE);
			AddTokenMap("DATEADD", SqlToken.DATEADD);
			AddTokenMap("DATEPART", SqlToken.DATEPART);
			AddTokenMap("DATEDIFF", SqlToken.DATEDIFF);
			AddTokenMap("DAY", SqlToken.DAY);
			AddTokenMap("EXP", SqlToken.EXP);
			AddTokenMap("FLOOR", SqlToken.FLOOR);
			AddTokenMap("GETDATE", SqlToken.GETDATE);
			AddTokenMap("LEN", SqlToken.LEN);
			AddTokenMap("LOG", SqlToken.LOG);
			AddTokenMap("MIN", SqlToken.MIN);
			AddTokenMap("MAX", SqlToken.MAX);
			AddTokenMap("MONTH", SqlToken.MONTH);
			AddTokenMap("ROUND", SqlToken.ROUND);
			AddTokenMap("RAISERROR", SqlToken.RAISERROR);
			AddTokenMap("REPLACE", SqlToken.REPLACE);
			AddTokenMap("SUM", SqlToken.SUM);
			AddTokenMap("SUSER_SNAME", SqlToken.SUSER_SNAME);
			AddTokenMap("SUBSTRING", SqlToken.SUBSTRING);
			AddTokenMap("YEAR", SqlToken.YEAR);

			AddTokenMap("ANSI_NULLS", SqlToken.ANSI_NULLS);
			AddTokenMap("ANSI_PADDING", SqlToken.ANSI_PADDING);
			AddTokenMap("ANSI_WARNINGS", SqlToken.ANSI_WARNINGS);
			AddTokenMap("ARITHABORT", SqlToken.ARITHABORT);
			AddTokenMap("COMMITTED", SqlToken.COMMITTED);
			AddTokenMap("CONNECT", SqlToken.CONNECT);
			AddTokenMap("COMMIT", SqlToken.Commit);
			AddTokenMap("CONCAT_NULL_YIELDS_NULL", SqlToken.CONCAT_NULL_YIELDS_NULL);
			AddTokenMap("DEADLOCK_PRIORITY", SqlToken.DEADLOCK_PRIORITY);
			AddTokenMap("HIGH", SqlToken.HIGH);
			AddTokenMap("HOLDLOCK", SqlToken.HOLDLOCK);
			AddTokenMap("ISOLATION", SqlToken.ISOLATION);
			AddTokenMap("IDENTITY_INSERT", SqlToken.IDENTITY_INSERT);
			AddTokenMap("LOW", SqlToken.LOW);
			AddTokenMap("LOWER", SqlToken.LOWER);
			AddTokenMap("LEVEL", SqlToken.LEVEL);
			AddTokenMap("LOCK_TIMEOUT", SqlToken.LOCK_TIMEOUT);
			AddTokenMap("NUMERIC_ROUNDABORT", SqlToken.NUMERIC_ROUNDABORT);
			AddTokenMap("NOEXEC", SqlToken.NOEXEC);
			AddTokenMap("NOLOCK", SqlToken.NOLOCK);
			AddTokenMap("NOCOUNT", SqlToken.NOCOUNT);
			AddTokenMap("NORMAL", SqlToken.NORMAL);
			AddTokenMap("NULLIF", SqlToken.NULLIF);
			AddTokenMap("NONCLUSTERED", SqlToken.NONCLUSTERED);
			AddTokenMap("MAXDOP", SqlToken.MAXDOP);
			AddTokenMap("QUOTED_IDENTIFIER", SqlToken.QUOTED_IDENTIFIER);
			AddTokenMap("ROWLOCK", SqlToken.ROWLOCK);
			AddTokenMap("READ", SqlToken.READ);
			AddTokenMap("REPEATABLE", SqlToken.REPEATABLE);
			AddTokenMap("ROLLBACK", SqlToken.ROLLBACK);
			AddTokenMap("SNAPSHOT", SqlToken.SNAPSHOT);
			AddTokenMap("SERIALIZABLE", SqlToken.SERIALIZABLE);
			AddTokenMap("TRANSACTION", SqlToken.TRANSACTION);
			AddTokenMap("TRAN", SqlToken.TRAN);
			AddTokenMap("UPDLOCK", SqlToken.UPDLOCK);
			AddTokenMap("UNCOMMITTED", SqlToken.UNCOMMITTED);
			AddTokenMap("XACT_ABORT", SqlToken.XACT_ABORT);

			AddSymbolMap("(", SqlToken.LParen);
			AddSymbolMap(")", SqlToken.RParen);
			AddSymbolMap(",", SqlToken.Comma);
			AddSymbolMap(";", SqlToken.Semicolon);
			AddSymbolMap(".", SqlToken.Dot);
			AddSymbolMap("=", SqlToken.Equal);
			AddSymbolMap("+", SqlToken.Plus);
			AddSymbolMap("+=", SqlToken.PlusEqual);
			AddSymbolMap("-=", SqlToken.MinusEqual);
			AddSymbolMap("-", SqlToken.Minus);
			AddSymbolMap("*", SqlToken.Asterisk);
			AddSymbolMap("/", SqlToken.Slash);
			AddSymbolMap("<>", SqlToken.NotEqual);
			AddSymbolMap("!=", SqlToken.NotEqual);
			AddSymbolMap("<", SqlToken.SmallerThan);
			AddSymbolMap("<=", SqlToken.SmallerThanOrEqual);
			AddSymbolMap(">", SqlToken.BiggerThan);
			AddSymbolMap(">=", SqlToken.BiggerThanOrEqual);
			AddSymbolMap("::", SqlToken.ColonColon);
			AddSymbolMap("&", SqlToken.Ampersand);
			AddSymbolMap("|", SqlToken.VerticalBar);
			AddSymbolMap("~", SqlToken.Tilde);
			AddSymbolMap("^", SqlToken.Caret);
		}

		protected override string GetTokenType(string token, string defaultTokenType)
		{
			return base.GetTokenType(token.ToUpper(), defaultTokenType);
		}

		protected override bool TryScanNext(TextSpan headSpan, out TextSpan tokenSpan)
		{
			tokenSpan = TextSpan.Empty;

			var head = GetSpanString(headSpan)[0];

			if (head == 'N' && TryNextChar('\'', out var head2))
			{
				headSpan = headSpan.Concat(head2);
				if (!TryRead(ReadQuoteString, headSpan, out var nstring))
				{
					ThrowHelper.ThrowScanException(this, $"Scan NString Error.");
				}
				nstring.Type = SqlToken.NString.ToString();
				tokenSpan = nstring;
				return true;
			}

			if (head == '0' && TryNextChar('x', out var hexHead))
			{
				headSpan = headSpan.Concat(hexHead);
				if (!TryRead(ReadHexNumber, headSpan, out var hexString))
				{
					ThrowHelper.ThrowScanException(this, $"Scan HexString Error.");
				}
				tokenSpan = hexString;
				return true;
			}

			if (char.IsDigit(head))
			{
				if (!TryRead(ReadNumber, headSpan, out var numberString))
				{
					ThrowHelper.ThrowScanException(this, $"Scan number Error.");
				}

				if (TryNextChar('.', out var floatHead))
				{
					headSpan = numberString.Concat(floatHead);
					if (!TryRead(ReadNumber, headSpan, out var floatString))
					{
						ThrowHelper.ThrowScanException(this, $"Scan float Error.");
					}
					tokenSpan = floatString;
					tokenSpan.Type = SqlToken.Number.ToString();
					return true;
				}

				tokenSpan = numberString;
				tokenSpan.Type = SqlToken.Number.ToString();
				return true;
			}

			if (head == '[' && TryRead(ReadSqlIdentifier, headSpan, out var sqlIdentifier))
			{
				tokenSpan = sqlIdentifier;
				return true;
			}

			var nextChar = PeekCh();
			if (head == ':' && char.IsLetter(nextChar) && TryRead(ReadIdentifier, headSpan, out var scriptIdentifier))
			{
				scriptIdentifier.Type = GetTokenType(scriptIdentifier, SqlToken.ScriptIdentifier);
				tokenSpan = scriptIdentifier;
				return true;
			}

			if (head == '/' && TryRead(ReadMultiComment, headSpan, out var multiComment))
			{
				tokenSpan = multiComment;
				return true;
			}

			if (head == '-' && TryNextChar('-', out var singleHead))
			{
				headSpan = headSpan.Concat(singleHead);
				if (!TryRead(ReadSingleComment, headSpan, out var singleComment))
				{
					ThrowHelper.ThrowScanException(this, $"Scan single comment Error.");
				}
				tokenSpan = singleComment;
				return true;
			}

			if (head == '\"' && TryRead(ReadDoubleQuoteString, headSpan, out var doubleQuoteString))
			{
				tokenSpan = doubleQuoteString;
				return true;
			}

			if (head == '\'' && TryRead(ReadQuoteString, headSpan, out var quoteString))
			{
				tokenSpan = quoteString;
				return true;
			}

			if (head == '@' && TryNextChar('@', out var atHead2))
			{
				headSpan = headSpan.Concat(atHead2);
				if (!TryRead(ReadIdentifier, headSpan, out var sysVariable))
				{
					throw new ScanException("Expect @@xxxx");
				}
				tokenSpan = sysVariable;
				tokenSpan.Type = SqlToken.SystemVariable.ToString();
				return true;
			}

			if (head == '@' && TryRead(ReadIdentifier, headSpan, out var variable))
			{
				tokenSpan = variable;
				tokenSpan.Type = SqlToken.Variable.ToString();
				return true;
			}

			if (head == '#' && TryRead(ReadIdentifier, headSpan, out var tmpTable))
			{
				tokenSpan = tmpTable;
				tokenSpan.Type = SqlToken.TempTable.ToString();
				return true;
			}

			if (_magnetHeadChars.Contains(head) && TryRead(ReadMagnetCompareSymbol, headSpan, out var magnetSymbol))
			{
				tokenSpan = magnetSymbol;
				return true;
			}

			return false;
		}

		private TextSpan ReadMagnetCompareSymbol(TextSpan head)
		{
			var startIndex = GetOffset();
			var index = 0;
			var sb = new StringBuilder();
			sb.Append(GetSpanString(head));
			do
			{
				var ch = PeekCh(index);
				if (ch == char.MinValue)
				{
					break;
				}
				if (!char.IsWhiteSpace(ch))
				{
					sb.Append(ch);
					break;
				}
				index++;
			} while (true);

			var tail = new TextSpan
			{
				Offset = startIndex,
				Length = index + 1
			};

			var peekSymbol = sb.ToString();
			if (index > 0 && _symbolToTokenTypeMap.ContainsKey(peekSymbol))
			{
				SetOffset(tail.Offset + tail.Length);
				var span = head.Concat(tail);
				span.Type = _symbolToTokenTypeMap[peekSymbol];
				return span;
			}

			return TextSpan.Empty;
		}

		private string GetTokenType(TextSpan span, SqlToken defaultTokenType)
		{
			var tokenStr = GetSpanString(span);
			return GetTokenType(tokenStr, defaultTokenType.ToString());
		}

		protected TextSpan ReadMultiComment(TextSpan head)
		{
			if (PeekCh() != '*')
			{
				return TextSpan.Empty;
			}

			var content = ReadUntil(head, ch =>
			{
				if (ch != '*')
				{
					return true;
				}
				if (PeekCh(1) == '/')
				{
					return false;
				}
				return true;
			});

			var tail = ConsumeCharacters("*/");
			content = content.Concat(tail);
			content.Type = SqlToken.MultiComment.ToString();
			return content;
		}

		protected TextSpan ReadSingleComment(TextSpan head)
		{
			var content = ReadUntil(head, ch =>
			{
				return ch != '\r';
			});

			content.Type = SqlToken.SingleComment.ToString();
			return content;
		}

		protected TextSpan ReadDoubleQuoteString(TextSpan head)
		{
			var content = ReadUntil(head, ch =>
			{
				return ch != '"';
			});

			var tail = ConsumeCharacters("\"");
			content = content.Concat(tail);
			content.Type = SqlToken.DoubleQuoteString.ToString();
			return content;
		}

		protected TextSpan ReadQuoteString(TextSpan head)
		{
			var content = head;
			do
			{
				var charSpan = PeekSpan();
				if (charSpan.IsEmpty)
				{
					break;
				}
				var currChar = PeekCh();
				var nextChar = PeekCh(1);
				if (currChar == '\'' && nextChar == '\'')
				{
					var nextCharSpan = PeekSpan(1);
					content = content.Concat(charSpan);
					content = content.Concat(nextCharSpan);
					NextChar();
					NextChar();
					continue;
				}
				if (currChar == '\'')
				{
					break;
				}
				content = content.Concat(charSpan);
				NextChar();
			} while (true);


			var tail = ConsumeCharacters("'");
			content = content.Concat(tail);
			content.Type = SqlToken.QuoteString.ToString();
			return content;
		}

		protected TextSpan ReadSqlIdentifier(TextSpan head)
		{
			if (PeekCh() == ']')
			{
				return TextSpan.Empty;
			}

			var content = ReadUntil(head, ch =>
			{
				return ch != ']';
			});

			var tail = ConsumeCharacters("]");

			content = content.Concat(tail);
			content.Type = SqlToken.SqlIdentifier.ToString();
			return content;
		}
	}
}
