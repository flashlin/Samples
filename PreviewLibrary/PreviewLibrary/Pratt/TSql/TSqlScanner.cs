using PreviewLibrary.Pratt.Core;
using System;
using System.Linq;
using System.Text;

namespace PreviewLibrary.Pratt.TSql
{
	public class TSqlScanner : StringScanner
	{
		public TSqlScanner(string text)
			: base(text)
		{
			AddToken("AS", SqlToken.As);
			AddToken("AND", SqlToken.And);
			AddToken("BREAK", SqlToken.Break);
			AddToken("BEGIN", SqlToken.Begin);
			AddToken("BIT", SqlToken.Bit);
			AddToken("CAST", SqlToken.Cast);
			AddToken("CASE", SqlToken.Case);
			AddToken("CREATE", SqlToken.Create);
			AddToken("DECLARE", SqlToken.Declare);
			AddToken("DELETE", SqlToken.Delete);
			AddToken("DEFAULT", SqlToken.Default);
			AddToken("DATETIME", SqlToken.DateTime);
			AddToken("DATETIME2", SqlToken.DateTime2);
			AddToken("DECIMAL", SqlToken.Decimal);
			AddToken("ERROR", SqlToken.Error);
			AddToken("EXIT", SqlToken.Exit);
			AddToken("END", SqlToken.End);
			AddToken("EXISTS", SqlToken.Exists);
			AddToken("EXECUTE", SqlToken.Execute);
			AddToken("EXEC", SqlToken.Exec);
			AddToken("ELSE", SqlToken.Else);
			AddToken("FROM", SqlToken.From);
			AddToken("FUNCTION", SqlToken.Function);
			AddToken("GRANT", SqlToken.Grant);
			AddToken("GO", SqlToken.Go);
			AddToken("IF", SqlToken.If);
			AddToken("IS", SqlToken.Is);
			AddToken("INSERT", SqlToken.Insert);
			AddToken("INTO", SqlToken.Into);
			AddToken("INT", SqlToken.Int);
			AddToken("LIKE", SqlToken.Like);
			AddToken("NVARCHAR", SqlToken.NVarchar);
			AddToken("NOT", SqlToken.Not);
			AddToken("NUMERIC", SqlToken.Numeric);
			AddToken("ON", SqlToken.On);
			AddToken("OBJECT", SqlToken.Object);
			AddToken("OFF", SqlToken.Off);
			AddToken("OR", SqlToken.Or);
			AddToken("PROCEDURE", SqlToken.Procedure);
			AddToken("RETURNS", SqlToken.Returns);
			AddToken("SET", SqlToken.Set);
			AddToken("SELECT", SqlToken.Select);
			AddToken("SMALLDATETIME", SqlToken.SmallDateTime);
			AddToken("TO", SqlToken.To);
			AddToken("TOP", SqlToken.Top);
			AddToken("TABLE", SqlToken.Table);
			AddToken("THEN", SqlToken.Then);
			AddToken("UPDATE", SqlToken.Update);
			AddToken("VALUES", SqlToken.Values);
			AddToken("VARCHAR", SqlToken.Varchar);
			AddToken("WHERE", SqlToken.Where);
			AddToken("WITH", SqlToken.With);
			AddToken("WHEN", SqlToken.When);
			AddToken(":SETVAR", SqlToken.ScriptSetVar);
			AddToken(":ON", SqlToken.ScriptOn);

			AddToken("ANSI_NULLS", SqlToken.ANSI_NULLS);
			AddToken("ANSI_PADDING", SqlToken.ANSI_PADDING);
			AddToken("ANSI_WARNINGS", SqlToken.ANSI_WARNINGS);
			AddToken("ARITHABORT", SqlToken.ARITHABORT);
			AddToken("CONNECT", SqlToken.CONNECT);
			AddToken("CONCAT_NULL_YIELDS_NULL", SqlToken.CONCAT_NULL_YIELDS_NULL);
			AddToken("QUOTED_IDENTIFIER", SqlToken.QUOTED_IDENTIFIER);
			AddToken("NUMERIC_ROUNDABORT", SqlToken.NUMERIC_ROUNDABORT);
			AddToken("NOEXEC", SqlToken.NOEXEC);
			AddToken("NOLOCK", SqlToken.NOLOCK);
			AddToken("IDENTITY_INSERT", SqlToken.IDENTITY_INSERT);

			AddSymbol("(", SqlToken.LParen);
			AddSymbol(")", SqlToken.RParen);
			AddSymbol(",", SqlToken.Comma);
			AddSymbol(";", SqlToken.Semicolon);
			AddSymbol(".", SqlToken.Dot);
			AddSymbol("=", SqlToken.Equal);
			AddSymbol("+", SqlToken.Plus);
			AddSymbol("-", SqlToken.Minus);
			AddSymbol("*", SqlToken.Asterisk);
			AddSymbol("/", SqlToken.Slash);
			AddSymbol("<>", SqlToken.SmallerBiggerThan);
		}

		protected void AddToken(string token, SqlToken tokenType)
		{
			AddTokenMap(token.ToUpper(), tokenType.ToString());
		}

		protected void AddSymbol(string symbol, SqlToken tokenType)
		{
			AddSymbolMap(symbol, tokenType.ToString());
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

			if (head == ':' && TryNextChar(':', out var colonHead))
			{
				tokenSpan = tokenSpan.Concat(colonHead);
				tokenSpan.Type = SqlToken.ColonColon.ToString();
				return true;
			}

			if (head == ':' && TryRead(ReadIdentifier, headSpan, out var scriptIdentifier))
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

			if (TryRead(ReadSymbol, headSpan, out var symbol))
			{
				tokenSpan = symbol;
				return true;
			}

			return false;
		}

		protected TextSpan ReadSymbol(TextSpan headSpan)
		{
			var maxSymbolLen = _symbolToTokenTypeMap.Keys.OrderByDescending(x => x.Length).First().Length;
			for (var tailLen = maxSymbolLen - 1; tailLen >= 0; tailLen--)
			{
				var tailSpan = PeekSpan(0, tailLen);
				var span = headSpan.Concat(tailSpan);
				var spanStr = GetSpanString(span);
				if (_symbolToTokenTypeMap.TryGetValue(spanStr, out var symbolType))
				{
					span.Type = symbolType;
					return span;
				}
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
