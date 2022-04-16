using PreviewLibrary.Pratt.Core;
using System;
using System.Linq;
using System.Text;

namespace PreviewLibrary.Pratt.TSql
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
			AddTokenMap("AND", SqlToken.And);
			AddTokenMap("ALL", SqlToken.All);
			AddTokenMap("BREAK", SqlToken.Break);
			AddTokenMap("BEGIN", SqlToken.Begin);
			AddTokenMap("BIT", SqlToken.Bit);
			AddTokenMap("CONTINUE", SqlToken.Continue);
			AddTokenMap("CHAR", SqlToken.Char);
			AddTokenMap("CAST", SqlToken.Cast);
			AddTokenMap("CASE", SqlToken.Case);
			AddTokenMap("CROSS", SqlToken.Cross);
			AddTokenMap("CREATE", SqlToken.Create);
			AddTokenMap("CONVERT", SqlToken.Convert);
			AddTokenMap("DECLARE", SqlToken.Declare);
			AddTokenMap("DELETE", SqlToken.Delete);
			AddTokenMap("DEFAULT", SqlToken.Default);
			AddTokenMap("DATETIME", SqlToken.DateTime);
			AddTokenMap("DATETIME2", SqlToken.DateTime2);
			AddTokenMap("DECIMAL", SqlToken.Decimal);
			AddTokenMap("ERROR", SqlToken.Error);
			AddTokenMap("EXIT", SqlToken.Exit);
			AddTokenMap("END", SqlToken.End);
			AddTokenMap("EXISTS", SqlToken.Exists);
			AddTokenMap("EXECUTE", SqlToken.Execute);
			AddTokenMap("EXEC", SqlToken.Exec);
			AddTokenMap("ELSE", SqlToken.Else);
			AddTokenMap("FROM", SqlToken.From);
			AddTokenMap("FLOAT", SqlToken.Float);
			AddTokenMap("FULL", SqlToken.Full);
			AddTokenMap("FUNCTION", SqlToken.Function);
			AddTokenMap("GRANT", SqlToken.Grant);
			AddTokenMap("GO", SqlToken.Go);
			AddTokenMap("IF", SqlToken.If);
			AddTokenMap("IS", SqlToken.Is);
			AddTokenMap("IN", SqlToken.In);
			AddTokenMap("INSERT", SqlToken.Insert);
			AddTokenMap("INTO", SqlToken.Into);
			AddTokenMap("INT", SqlToken.Int);
			AddTokenMap("INNER", SqlToken.Inner);
			AddTokenMap("JOIN", SqlToken.Join);
			AddTokenMap("KEY", SqlToken.Key);
			AddTokenMap("LIKE", SqlToken.Like);
			AddTokenMap("LEFT", SqlToken.Left);
			AddTokenMap("NVARCHAR", SqlToken.NVarchar);
			AddTokenMap("NOT", SqlToken.Not);
			AddTokenMap("NUMERIC", SqlToken.Numeric);
			AddTokenMap("MAX", SqlToken.Max);
			AddTokenMap("ON", SqlToken.On);
			AddTokenMap("OBJECT", SqlToken.Object);
			AddTokenMap("OFF", SqlToken.Off);
			AddTokenMap("OR", SqlToken.Or);
			AddTokenMap("OUTER", SqlToken.Outer);
			AddTokenMap("PRIMARY", SqlToken.Primary);
			AddTokenMap("PROCEDURE", SqlToken.Procedure);
			AddTokenMap("RETURNS", SqlToken.Returns);
			AddTokenMap("RIGHT", SqlToken.Right);
			AddTokenMap("SET", SqlToken.Set);
			AddTokenMap("SELECT", SqlToken.Select);
			AddTokenMap("SMALLDATETIME", SqlToken.SmallDateTime);
			AddTokenMap("TO", SqlToken.To);
			AddTokenMap("TOP", SqlToken.Top);
			AddTokenMap("TABLE", SqlToken.Table);
			AddTokenMap("THEN", SqlToken.Then);
			AddTokenMap("TINYINT", SqlToken.TinyInt);
			AddTokenMap("UPDATE", SqlToken.Update);
			AddTokenMap("UNION", SqlToken.Union);
			AddTokenMap("VALUES", SqlToken.Values);
			AddTokenMap("VARCHAR", SqlToken.Varchar);
			AddTokenMap("WHERE", SqlToken.Where);
			AddTokenMap("WITH", SqlToken.With);
			AddTokenMap("WHEN", SqlToken.When);
			AddTokenMap("WHILE", SqlToken.While);
			AddTokenMap(":SETVAR", SqlToken.ScriptSetVar);
			AddTokenMap(":ON", SqlToken.ScriptOn);

			AddTokenMap("ANSI_NULLS", SqlToken.ANSI_NULLS);
			AddTokenMap("ANSI_PADDING", SqlToken.ANSI_PADDING);
			AddTokenMap("ANSI_WARNINGS", SqlToken.ANSI_WARNINGS);
			AddTokenMap("ARITHABORT", SqlToken.ARITHABORT);
			AddTokenMap("CONNECT", SqlToken.CONNECT);
			AddTokenMap("CONCAT_NULL_YIELDS_NULL", SqlToken.CONCAT_NULL_YIELDS_NULL);
			AddTokenMap("QUOTED_IDENTIFIER", SqlToken.QUOTED_IDENTIFIER);
			AddTokenMap("NUMERIC_ROUNDABORT", SqlToken.NUMERIC_ROUNDABORT);
			AddTokenMap("NOEXEC", SqlToken.NOEXEC);
			AddTokenMap("NOLOCK", SqlToken.NOLOCK);
			AddTokenMap("IDENTITY_INSERT", SqlToken.IDENTITY_INSERT);

			AddSymbolMap("(", SqlToken.LParen);
			AddSymbolMap(")", SqlToken.RParen);
			AddSymbolMap(",", SqlToken.Comma);
			AddSymbolMap(";", SqlToken.Semicolon);
			AddSymbolMap(".", SqlToken.Dot);
			AddSymbolMap("=", SqlToken.Equal);
			AddSymbolMap("+", SqlToken.Plus);
			AddSymbolMap("-", SqlToken.Minus);
			AddSymbolMap("*", SqlToken.Asterisk);
			AddSymbolMap("/", SqlToken.Slash);
			AddSymbolMap("<>", SqlToken.SmallerBiggerThan);
			AddSymbolMap("<", SqlToken.SmallerThan);
			AddSymbolMap("<=", SqlToken.SmallerThanOrEqual);
			AddSymbolMap(">", SqlToken.BiggerThan);
			AddSymbolMap(">=", SqlToken.BiggerThanOrEqual);
			AddSymbolMap("::", SqlToken.ColonColon);
			AddSymbolMap("&", SqlToken.Ampersand);
			AddSymbolMap("|", SqlToken.VerticalBar);
			AddSymbolMap("~", SqlToken.Tilde);
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

			if (head == '@' && TryRead(ReadIdentifier, headSpan, out var variable))
			{
				tokenSpan = variable;
				tokenSpan.Type = SqlToken.Variable.ToString();
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
				if (ch == Char.MinValue)
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
				return (ch != '\r');
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
