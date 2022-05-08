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
			AddTokenMap("ALL", SqlToken.ALL);
			AddTokenMap("BREAK", SqlToken.Break);
			AddTokenMap("BEGIN", SqlToken.Begin);
			AddTokenMap("BIT", SqlToken.Bit);
			AddTokenMap("BY", SqlToken.By);
			AddTokenMap("BETWEEN", SqlToken.Between);
			AddTokenMap("BIGINT", SqlToken.Bigint);
			AddTokenMap("CONTINUE", SqlToken.Continue);
			AddTokenMap("CHAR", SqlToken.CHAR);
			AddTokenMap("CAST", SqlToken.Cast);
			AddTokenMap("CASE", SqlToken.Case);
			AddTokenMap("CLOSE", SqlToken.Close);
			AddTokenMap("CURSOR", SqlToken.Cursor);
			AddTokenMap("CROSS", SqlToken.Cross);
			AddTokenMap("CREATE", SqlToken.Create);
			AddTokenMap("CONVERT", SqlToken.Convert);
			AddTokenMap("DBCC", SqlToken.Dbcc);
			AddTokenMap("DEALLOCATE", SqlToken.Deallocate);
			AddTokenMap("DROP", SqlToken.Drop);
			AddTokenMap("DENSE_RANK", SqlToken.DENSE_RANK);
			AddTokenMap("DESC", SqlToken.Desc);
			AddTokenMap("DISTINCT", SqlToken.Distinct);
			AddTokenMap("DECLARE", SqlToken.Declare);
			AddTokenMap("DELETE", SqlToken.Delete);
			AddTokenMap("DEFAULT", SqlToken.Default);
			AddTokenMap("EXPLICIT", SqlToken.EXPLICIT);
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
			AddTokenMap("FOR", SqlToken.FOR);
			AddTokenMap("FROM", SqlToken.From);
			AddTokenMap("FLOAT", SqlToken.Float);
			AddTokenMap("FULL", SqlToken.Full);
			AddTokenMap("FETCH", SqlToken.Fetch);
			AddTokenMap("FUNCTION", SqlToken.Function);
			AddTokenMap("GRANT", SqlToken.Grant);
			AddTokenMap("GO", SqlToken.Go);
			AddTokenMap("GROUP", SqlToken.Group);
			AddTokenMap("HAVING", SqlToken.Having);
			AddTokenMap("IF", SqlToken.If);
			AddTokenMap("IS", SqlToken.Is);
			AddTokenMap("IN", SqlToken.In);
			AddTokenMap("INT", SqlToken.Int);
			AddTokenMap("INSERT", SqlToken.Insert);
			AddTokenMap("INTO", SqlToken.Into);
			AddTokenMap("INDEX", SqlToken.Index);
			AddTokenMap("INNER", SqlToken.Inner);
			AddTokenMap("INSERTED", SqlToken.Inserted);
			AddTokenMap("JOIN", SqlToken.Join);
			AddTokenMap("LIKE", SqlToken.Like);
			AddTokenMap("ALTER", SqlToken.Alter);
			AddTokenMap("LEFT", SqlToken.Left);
			AddTokenMap("NVARCHAR", SqlToken.NVarchar);
			AddTokenMap("NEXT", SqlToken.Next);
			AddTokenMap("NOT", SqlToken.Not);
			AddTokenMap("NULL", SqlToken.Null);
			AddTokenMap("NUMERIC", SqlToken.Numeric);
			AddTokenMap("MERGE", SqlToken.Merge);
			AddTokenMap("MATCHED", SqlToken.Matched);
			AddTokenMap("OBJECT", SqlToken.Object);
			AddTokenMap("OUT", SqlToken.Out);
			AddTokenMap("OR", SqlToken.Or);
			AddTokenMap("OPEN", SqlToken.Open);
			AddTokenMap("ORDER", SqlToken.Order);
			AddTokenMap("OUTER", SqlToken.Outer);
			AddTokenMap("OVER", SqlToken.Over);
			AddTokenMap("OUTPUT", SqlToken.Output);
			AddTokenMap("OPTION", SqlToken.Option);
			AddTokenMap("PARTITION", SqlToken.Partition);
			AddTokenMap("PIVOT", SqlToken.Pivot);
			AddTokenMap("PROCEDURE", SqlToken.Procedure);
			AddTokenMap("PROC", SqlToken.Procedure);
			AddTokenMap("PATH", SqlToken.PATH);
			AddTokenMap("ROW_NUMBER", SqlToken.ROW_NUMBER);
			AddTokenMap("REBUILD", SqlToken.Rebuild);
			AddTokenMap("PRINT", SqlToken.Print);
			AddTokenMap("RANK", SqlToken.Rank);
			AddTokenMap("RAW", SqlToken.RAW);
			AddTokenMap("ROOT", SqlToken.ROOT);
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
			AddTokenMap("TABLE", SqlToken.TABLE);
			AddTokenMap("THEN", SqlToken.Then);
			AddTokenMap("TINYINT", SqlToken.TinyInt);
			AddTokenMap("UNPIVOT", SqlToken.UnPivot);
			AddTokenMap("UPDATEUSAGE", SqlToken.UpdateUsage);
			AddTokenMap("UPDATE", SqlToken.Update);
			AddTokenMap("USING", SqlToken.Using);
			AddTokenMap("UNION", SqlToken.Union);
			AddTokenMap("VALUES", SqlToken.Values);
			AddTokenMap("VARCHAR", SqlToken.Varchar);
			AddTokenMap("WHERE", SqlToken.Where);
			AddTokenMap("WITH", SqlToken.With);
			AddTokenMap("WHEN", SqlToken.When);
			AddTokenMap("WHILE", SqlToken.While);
			AddTokenMap("XML", SqlToken.XML);
			AddTokenMap(":SETVAR", SqlToken.ScriptSetVar);
			AddTokenMap(":ON", SqlToken.ScriptOn);
			
			AddFuncNameMap("ABS", SqlToken.ABS);
			AddFuncNameMap("CHARINDEX", SqlToken.CHARINDEX);
			AddFuncNameMap("COUNT", SqlToken.COUNT);
			AddFuncNameMap("CLUSTERED", SqlToken.CLUSTERED);
			AddFuncNameMap("COALESCE", SqlToken.COALESCE);
			AddFuncNameMap("DATEADD", SqlToken.DATEADD);
			AddFuncNameMap("DATEPART", SqlToken.DATEPART);
			AddFuncNameMap("DATEDIFF", SqlToken.DATEDIFF);
			AddFuncNameMap("DAY", SqlToken.DAY);
			AddFuncNameMap("EXP", SqlToken.EXP);
			AddFuncNameMap("FLOOR", SqlToken.FLOOR);
			AddFuncNameMap("GETDATE", SqlToken.GETDATE);
			AddFuncNameMap("ISNULL", SqlToken.ISNULL);
			AddFuncNameMap("LEN", SqlToken.LEN);
			AddFuncNameMap("LOG", SqlToken.LOG);
			AddFuncNameMap("NEWID", SqlToken.NEWID);
			AddFuncNameMap("NODES", SqlToken.NODES);
			AddFuncNameMap("MIN", SqlToken.MIN);
			AddFuncNameMap("MAX", SqlToken.MAX);
			AddFuncNameMap("MONTH", SqlToken.MONTH);
			AddFuncNameMap("ROUND", SqlToken.ROUND);
			AddFuncNameMap("RAISERROR", SqlToken.RAISERROR);
			AddFuncNameMap("REPLACE", SqlToken.REPLACE);
			AddFuncNameMap("SUM", SqlToken.SUM);
			AddFuncNameMap("SUSER_SNAME", SqlToken.SUSER_SNAME);
			AddFuncNameMap("SUBSTRING", SqlToken.SUBSTRING);
			AddFuncNameMap("YEAR", SqlToken.YEAR);

			AddTokenMap("ADD", SqlToken.ADD);
			AddTokenMap("AFTER", SqlToken.AFTER);
			AddTokenMap("APPLY", SqlToken.APPLY);
			AddTokenMap("AUTO", SqlToken.AUTO);
			AddTokenMap("ANSI_NULLS", SqlToken.ANSI_NULLS);
			AddTokenMap("ANSI_PADDING", SqlToken.ANSI_PADDING);
			AddTokenMap("ANSI_WARNINGS", SqlToken.ANSI_WARNINGS);
			AddTokenMap("ARITHABORT", SqlToken.ARITHABORT);
			AddTokenMap("ALLOW_ROW_LOCKS", SqlToken.ALLOW_ROW_LOCKS);
			AddTokenMap("ALLOW_PAGE_LOCKS", SqlToken.ALLOW_PAGE_LOCKS);
			AddTokenMap("AUTHORIZATION", SqlToken.AUTHORIZATION);
			AddTokenMap("CHECK", SqlToken.CHECK);
			AddTokenMap("CALLER", SqlToken.CALLER);
			AddTokenMap("CONSTRAINT", SqlToken.CONSTRAINT);
			AddTokenMap("COMMITTED", SqlToken.COMMITTED);
			AddTokenMap("CONNECT", SqlToken.CONNECT);
			AddTokenMap("COMMIT", SqlToken.Commit);
			AddTokenMap("CHECKIDENT", SqlToken.CHECKIDENT);
			AddTokenMap("COUNT_ROWS", SqlToken.COUNT_ROWS);
			AddTokenMap("CURRENT", SqlToken.CURRENT);
			AddTokenMap("CONCAT_NULL_YIELDS_NULL", SqlToken.CONCAT_NULL_YIELDS_NULL);
			AddTokenMap("DEADLOCK_PRIORITY", SqlToken.DEADLOCK_PRIORITY);
			AddTokenMap("DATABASE", SqlToken.DATABASE);
			AddTokenMap("DEFAULT_SCHEMA", SqlToken.DEFAULT_SCHEMA);
			AddTokenMap("DISABLE", SqlToken.DISABLE);
			AddTokenMap("DYNAMIC", SqlToken.DYNAMIC);
			AddTokenMap("DROP_EXISTING", SqlToken.DROP_EXISTING);
			AddTokenMap("ENABLE", SqlToken.ENABLE);
			AddTokenMap("FILEGROUP", SqlToken.FILEGROUP);
			AddTokenMap("FORCESEEK", SqlToken.FORCESEEK);
			AddTokenMap("FILLFACTOR", SqlToken.FILLFACTOR);
			AddTokenMap("FORWARD_ONLY", SqlToken.FORWARD_ONLY);
			AddTokenMap("FAST_FORWARD", SqlToken.FAST_FORWARD);
			AddTokenMap("GLOBAL", SqlToken.GLOBAL);
			AddTokenMap("HIGH", SqlToken.HIGH);
			AddTokenMap("HOLDLOCK", SqlToken.HOLDLOCK);
			AddTokenMap("INPUTBUFFER", SqlToken.INPUTBUFFER);
			AddTokenMap("ISOLATION", SqlToken.ISOLATION);
			AddTokenMap("IDENTITY_INSERT", SqlToken.IDENTITY_INSERT);
			AddTokenMap("IDENTITY", SqlToken.IDENTITY);
			AddTokenMap("IGNORE_DUP_KEY", SqlToken.IGNORE_DUP_KEY);
			AddTokenMap("KEY", SqlToken.KEY);
			AddTokenMap("KEYSET", SqlToken.KEYSET);
			AddTokenMap("LOCK_ESCALATION", SqlToken.LOCK_ESCALATION);
			AddTokenMap("LOW", SqlToken.LOW);
			AddTokenMap("LOCAL", SqlToken.LOCAL);
			AddTokenMap("LOGIN", SqlToken.LOGIN);
			AddTokenMap("LOWER", SqlToken.LOWER);
			AddTokenMap("LEVEL", SqlToken.LEVEL);
			AddTokenMap("LOGINFO", SqlToken.LOGINFO);
			AddTokenMap("LOCK_TIMEOUT", SqlToken.LOCK_TIMEOUT);
			AddTokenMap("NOCHECK", SqlToken.NOCHECK);
			AddTokenMap("NUMERIC_ROUNDABORT", SqlToken.NUMERIC_ROUNDABORT);
			AddTokenMap("NO_INFOMSGS", SqlToken.NO_INFOMSGS);
			AddTokenMap("NOEXEC", SqlToken.NOEXEC);
			AddTokenMap("NORESEED", SqlToken.NORESEED);
			AddTokenMap("NOLOCK", SqlToken.NOLOCK);
			AddTokenMap("NOCOUNT", SqlToken.NOCOUNT);
			AddTokenMap("NORMAL", SqlToken.NORMAL);
			AddTokenMap("NULLIF", SqlToken.NULLIF);
			AddTokenMap("NOWAIT", SqlToken.NOWAIT);
			AddTokenMap("NONCLUSTERED", SqlToken.NONCLUSTERED);
			AddTokenMap("MAXDOP", SqlToken.MAXDOP);
			AddTokenMap("OBJECT_NAME", SqlToken.OBJECT_NAME);
			AddTokenMap("OWNER", SqlToken.OWNER);
			AddTokenMap("OFF", SqlToken.OFF);
			AddTokenMap("ON", SqlToken.ON);
			AddTokenMap("ONLINE", SqlToken.ONLINE);
			AddTokenMap("OPTIMISTIC", SqlToken.OPTIMISTIC);
			AddTokenMap("PASSWORD", SqlToken.PASSWORD);
			AddTokenMap("PRIMARY", SqlToken.PRIMARY);
			AddTokenMap("PAD_INDEX", SqlToken.PAD_INDEX);
			AddTokenMap("PERCENT", SqlToken.PERCENT);
			AddTokenMap("QUOTED_IDENTIFIER", SqlToken.QUOTED_IDENTIFIER);
			AddTokenMap("ROLE", SqlToken.ROLE);
			AddTokenMap("READ_ONLY", SqlToken.READ_ONLY);
			AddTokenMap("RESEED", SqlToken.RESEED);
			AddTokenMap("REPLICATION", SqlToken.REPLICATION);
			AddTokenMap("ROWLOCK", SqlToken.ROWLOCK);
			AddTokenMap("REORGANIZE", SqlToken.REORGANIZE);
			AddTokenMap("READ", SqlToken.READ);
			AddTokenMap("REPEATABLE", SqlToken.REPEATABLE);
			AddTokenMap("ROLLBACK", SqlToken.ROLLBACK);
			AddTokenMap("STATISTICS_NORECOMPUTE", SqlToken.STATISTICS_NORECOMPUTE);
			AddTokenMap("SNAPSHOT", SqlToken.SNAPSHOT);
			AddTokenMap("SCROLL", SqlToken.SCROLL);
			AddTokenMap("SCROLL_LOCKS", SqlToken.SCROLL_LOCKS);
			AddTokenMap("SELF", SqlToken.SELF);
			AddTokenMap("SQLPERF", SqlToken.SQLPERF);
			AddTokenMap("STATIC", SqlToken.STATIC);
			AddTokenMap("SYNONYM", SqlToken.SYNONYM);
			AddTokenMap("SERVER", SqlToken.SERVER);
			AddTokenMap("SHRINKFILE", SqlToken.SHRINKFILE);
			AddTokenMap("SORT_IN_TEMPDB", SqlToken.SORT_IN_TEMPDB);
			AddTokenMap("SERIALIZABLE", SqlToken.SERIALIZABLE);
			AddTokenMap("TRANSACTION", SqlToken.TRANSACTION);
			AddTokenMap("TRIGGER", SqlToken.TRIGGER);
			AddTokenMap("TRAN", SqlToken.TRAN);
			AddTokenMap("TYPE", SqlToken.TYPE);
			AddTokenMap("TYPE_WARNING", SqlToken.TYPE_WARNING);
			AddTokenMap("USER", SqlToken.USER);
			AddTokenMap("UPDLOCK", SqlToken.UPDLOCK);
			AddTokenMap("UNIQUE", SqlToken.UNIQUE);
			AddTokenMap("UNCOMMITTED", SqlToken.UNCOMMITTED);
			AddTokenMap("VIEW", SqlToken.VIEW);
			AddTokenMap("WITHOUT", SqlToken.WITHOUT);
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
			AddSymbolMap("%", SqlToken.Percent);
		}

		protected override string GetTokenType(string token, string defaultTokenType)
		{
			return base.GetTokenType(token.ToUpper(), defaultTokenType);
		}

		protected override bool TryScanNext(TextSpan headSpan, out TextSpan tokenSpan)
		{
			tokenSpan = TextSpan.Empty;

			var head = GetSpanString(headSpan)[0];

			if (head == ':' && TryNextChar('r', out var refSpan))
			{
				tokenSpan = headSpan.Concat(refSpan);
				tokenSpan = ReadBatchReferenceFile(tokenSpan);
				tokenSpan.Type = SqlToken.BatchRefFile.ToString();
				return true;
			}

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
			
			if (head == '$' && TryRead(ReadBatchVariable, headSpan, out var nameSpan)) 
			{
				tokenSpan = nameSpan;
				return true;
			}

			if (_magnetHeadChars.Contains(head) && TryRead(ReadMagnetCompareSymbol, headSpan, out var magnetSymbol))
			{
				tokenSpan = magnetSymbol;
				return true;
			}

			return false;
		}


		protected TextSpan ReadBatchVariable(TextSpan head)
		{
			if (!TryNextChar('(', out var lrent))
			{
				return TextSpan.Empty;
			}

			head = head.Concat(lrent);
			
			var identifier = ReadIdentifier(head);
			if (identifier.Length == 2)
			{
				return TextSpan.Empty;
			}
			
			if (!TryNextChar(')', out var rrent))
			{
				return TextSpan.Empty;
			}

			var token = identifier.Concat(rrent);
			token.Type = SqlToken.BatchVariable.ToString();
			return token;
		}

		private TextSpan ReadBatchReferenceFile(TextSpan head)
		{
			var content = ReadUntil(head, ch =>
			{
				return ch != '\n';
			});
			if (!Peek().IsEmpty)
			{
				ConsumeCharacters("\n");
			}
			return content;
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
				return ch != '\r' && ch != '\n';
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
