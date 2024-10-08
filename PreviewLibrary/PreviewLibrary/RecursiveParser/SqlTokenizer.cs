﻿using PreviewLibrary.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.RegularExpressions;
using T1.Standard.Extensions;

namespace PreviewLibrary.RecursiveParser
{
	public class SqlTokenizer : TokenizerBase
	{
		static string Word(string words)
		{
			var sb = new StringBuilder();
			foreach (var word in words)
			{
				if (char.IsLetter(word))
				{
					sb.Append($"[{char.ToLower(word)}{char.ToUpper(word)}]");
				}
				else if (char.IsDigit(word))
				{
					sb.Append(word);
				}
				else
				{
					sb.Append(Regex.Escape($"{word}"));
				}
			}
			return sb.ToString();
		}

		static readonly string PositiveInteger = @"\d+";
		static readonly string IntegerNumber = PositiveInteger;
		public static readonly string DecimalNumber = @"\d+\.\d*";
		static readonly string SqlIdent = @"\[[^\]]+\]";
		public static readonly string SqlVariable = @"\@([A-Za-z]|\d|_)+";
		public static readonly string SystemVariable = @"\@\@" + RegexPattern.Ident;
		public static readonly string MultiLineComment = "/\\*[^*]*\\*+(?:[^/*][^*]*\\*+)*/";
		public static readonly string SingleLineComment = @"--[^\r\n]*";
		static readonly string BatchInstruction = @"\:" + RegexPattern.Ident;
		public static readonly string DoubleQuotedString = @"""[^""]*""";
		public static readonly string QuotedString = @"N?'[^']*(?:''[^']*)*'";
		public static readonly string Hex16Number = "0x" + "[0-9a-fA-F]+";
		public static readonly string NOT_LIKE = Word("NOT") + $"{RegexPattern.Blank}+" + Word("LIKE");
		public static readonly string NOT_IN = Word("NOT") + $"{RegexPattern.Blank}+" + Word("IN");
		public static readonly string TEMP_TABLE_NAME = Word("#") + RegexPattern.Ident;

		public static readonly string[] _keywords = new[]
		{
			"ALL", "AND",  "AS",
			"BEGIN", "BY", "BETWEEN",
			"CASE", "CROSS",
			"DISTINCT",
			"END", "EXEC", "EXECUTE", "ELSE",
			"FROM", "FULL",
			"GROUP", "GRANT",
			"INNER", "INSERT", "INTO", "IF",
			"JOIN",
			"LEFT", "LIKE",

			"THEN", "TOP",
			"OR", "ON", "OUTER", "ORDER",
			"RIGHT", "RETURN",
			"SET", "SELECT",
			"UNION",
			"WHEN", "WHERE", "WITH",
			"MAX",
			"NULL", "NOT",
		};
		public static readonly string[] SqlFunc0Names = new[]
		{
			"DB_NAME", "GETDATE"
		};
		public static readonly string[] SqlFunc1Names = new[]
		{
			"COUNT",
			"EXISTS",
			"MAX",
			"SUM",
			"SUSER_SNAME",
		};
		public static readonly string[] SqlFunc2Names = new[]
		{
			"ISNULL", "COALESCE"
		};


		public static string[] DataTypes = new string[]
		{
			"BIT", "BIGINT",
			"CHAR",
			"DATETIME", "DATETIME2", "DECIMAL", "DATE",
			"FLOAT",
			"INT",
			"NUMERIC", "NVARCHAR",
			"SMALLDATETIME",
			"TINYINT", "TIMESTAMP",
			"VARCHAR",
		};

		private static readonly Dictionary<int, string[]> _sqlFuncArgsCount_SqlFuncNames = new Dictionary<int, string[]>()
		{
			{ 0, SqlFunc0Names },
			{ 1, SqlFunc1Names },
			{ 2, SqlFunc2Names },
		};

		public static string[] SqlFunctionNames =
			_sqlFuncArgsCount_SqlFuncNames.Values.SelectMany(x => x)
			.Concat(new[]
			{
				"CAST"
			}).ToArray();

		private static readonly Dictionary<string, int> _sqlFuncName_ArgsCount = new Dictionary<string, int>();

		static readonly string[] CompareOperSymbols = new[]
		{
			"!=", "<=", ">=", "<>",
			">", "<", "="
		};

		static readonly string[] CompareOperSymbolsContainsSpacesPattern =
			CompareOperSymbols.Where(x => x.Length > 1)
				.Select(SymbolContainsSpacesPattern)
				.ToArray();

		static string SymbolContainsSpacesPattern(string symbol)
		{
			var sb = new StringBuilder();
			for (var i = 0; i < symbol.Length; i++)
			{
				if (i != 0)
				{
					sb.Append(@"\s*");
				}
				sb.Append(Regex.Escape($"{symbol[i]}"));
			}
			return sb.ToString();
		}

		string PackSymbol(string text)
		{
			bool isMatch(string pattern)
			{
				return new Regex("^" + pattern + "$").IsMatch(text);
			}

			foreach (var pattern in CompareOperSymbolsContainsSpacesPattern)
			{
				if (isMatch(pattern))
				{
					return text.Replace(" ", "");
				}
			}
			return text.CondenseSpaces();
		}

		public static readonly string[] CompareOps = new[]
		{
			"LIKE", "IN", "IS",
		}.Concat(CompareOperSymbols).ToArray();

		public static readonly string[] ConcatOps = new[]
		{
			"AND", "OR"
		};

		static IEnumerable<string> Escape(IEnumerable<string> patterns)
		{
			return patterns.Select(e => Regex.Escape(e));
		}

		public static string[] Op1 = new[]
		{
			"*", "/"
		};

		public static string[] Op2 = new[]
		{
			"+", "-"
		};

		public static string[] Terms = new[]
		{
			"(", ")"
		};

		static readonly string[] OtherSymbols = new[]
		{
			"--", "\r\n", "::",
			"&", "|", ".", ",", "(", ")", "@", ";",
			"+", "-", "*", "/",
		}.Select(e => Regex.Escape(e)).ToArray();



		static readonly string[] AllStrings = new[]
		{
			QuotedString,
			DoubleQuotedString,
		};

		static readonly string[] AllPatterns =
			ConcatArray(
				CompareOperSymbolsContainsSpacesPattern,
				Escape(CompareOperSymbols),
				AllStrings,
				new[] {
					TEMP_TABLE_NAME,
					NOT_LIKE,
					NOT_IN,
					BatchInstruction,
					SingleLineComment,
					MultiLineComment,
					Hex16Number,
					DecimalNumber,
					IntegerNumber,
					SqlIdent,
					SystemVariable,
					SqlVariable,
					RegexPattern.Ident,
				},
				OtherSymbols)
			.ToArray();

		static string[] ConcatArray(params IEnumerable<string>[] patternsList)
		{
			var list = new List<string>();
			foreach (var patterns in patternsList)
			{
				list.AddRange(patterns);
			}
			return list.ToArray();
		}

		static SqlTokenizer()
		{
			foreach (var item in _sqlFuncArgsCount_SqlFuncNames)
			{
				var argsCount = item.Key;
				var names = item.Value;
				foreach (var name in names)
				{
					_sqlFuncName_ArgsCount[name] = argsCount;
				}
			}
		}

		public SqlTokenizer() : base(AllPatterns)
		{
		}

		protected override string GetCurrentText(string text)
		{
			return PackSymbol(text);
		}

		public bool IsIdentWord
		{
			get
			{
				char c = Ch;
				return char.IsLower(c) || char.IsUpper(c) || c == '_';
			}
		}

		private static HashSet<string> _reverseKeywords =
			_keywords.Concat(SqlFunctionNames).ToHashSet();

		private static bool IsReverseKeyword(string text)
		{
			return _reverseKeywords.Contains(text.ToUpper());
		}

		public bool IsIdent
		{
			get
			{
				//if (IsReverseKeyword(Text))
				//{
				//	return false;
				//}
				char c = Ch;
				return char.IsLower(c) || char.IsUpper(c) || c == '_';
			}
		}

		public bool IsSqlString
		{
			get
			{
				return IsMatchAny(AllStrings);
			}
		}

		public bool IsCompareOp
		{
			get
			{
				return Contains(CompareOps);
			}
		}

		public bool IsSqlIdent
		{
			get
			{
				if (IsReverseKeyword(Text))
				{
					return false;
				}
				if (Text.StartsWith("[") && Text.EndsWith("]"))
				{
					return true;
				}
				//if (Text.StartsWith("@"))
				//{
				//	return true;
				//}
				if (Text.StartsWith("#"))
				{
					return true;
				}
				return IsIdent;
			}
		}

		public bool IsFuncName(out int argsCount)
		{
			if (_sqlFuncName_ArgsCount.ContainsKey(Text.ToUpper()))
			{
				argsCount = _sqlFuncName_ArgsCount[Text.ToUpper()];
				return true;
			}
			argsCount = -1;
			return false;
		}

		public bool Contains(string[] strs)
		{
			for (var i = 0; i < strs.Length; i++)
			{
				if (IgnoreCase(strs[i]))
				{
					return true;
				}
			}
			return false;
		}

		public string Read(string pattern, string patternName = null)
		{
			if (!IsMatch(pattern))
			{
				var name = pattern;
				if (!string.IsNullOrEmpty(patternName))
				{
					name = patternName;
				}
				throw new Exception($"Expect match '{name}' pattern, but got '{Text}'");
			}
			var text = Text;
			Move();
			return text;
		}

		public bool TryMatch(string pattern, out string str)
		{
			var success = IsMatch(pattern);
			if (success)
			{
				str = Text;
				Move();
				return success;
			}
			str = null;
			return success;
		}

		public override bool Move()
		{
			bool isComment()
			{
				if (IsMatch(SingleLineComment)) return true;
				if (IsMatch(MultiLineComment)) return true;
				return false;
			}

			var success = false;
			do
			{
				success = base.Move();
			} while (success && isComment());
			return success;
		}

		public bool IsMatchAny(params string[] patterns)
		{
			for (var i = 0; i < patterns.Length; i++)
			{
				var pattern = patterns[i];
				if (IsMatch(pattern))
				{
					return true;
				}
			}
			return false;
		}

		public bool TryEqual(string[] keywords, out string token)
		{
			if (keywords.Any(x => IgnoreCase(x)))
			{
				token = Text;
				Move();
				return true;
			}
			token = null;
			return false;
		}
	}
}
