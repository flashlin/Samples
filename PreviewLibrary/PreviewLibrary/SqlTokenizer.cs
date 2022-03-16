using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using T1.Standard.Extensions;

namespace PreviewLibrary
{
	public class SqlTokenizer : TokenizerBase
	{
		static readonly string PositiveInteger = @"\d+";
		static readonly string IntegerNumber = @"-?" + PositiveInteger;
		public static readonly string DecimalNumber = @"[-]?\d+\.\d*";
		static readonly string SqlIdent = @"\[[^\]]+\]";
		public static readonly string SqlVariable = @"\@" + RegexPattern.Ident;
		public static readonly string MultiLineComment = "/\\*[^*]*\\*+(?:[^/*][^*]*\\*+)*/";
		public static readonly string SingleLineComment = @"--[^\r\n]*";
		static readonly string BatchInstruction = @"\:" + RegexPattern.Ident;
		public static readonly string DoubleQuotedString = @"""[^""]*""";
		public static readonly string QuotedString = @"N?'[^']*(?:''[^']*)*'";
		public static readonly string Hex16Number = "0x" + "[0-9a-fA-F]+";
		public static readonly string[] _keywords = new[]
		{
			"SELECT", "FROM", "WHERE", "AS", "WITH",
			"AND", "OR", "BEGIN", "END", "EXEC", "EXECUTE", "NULL"
		};
		public static readonly string[] SqlFunc0Names = new[]
		{
			"DB_NAME", "GETDATE"
		};
		public static readonly string[] SqlFunc1Names = new[]
		{
			"EXISTS", "SUSER_SNAME"
		};
		public static readonly string[] SqlFunc2Names = new[]
		{
			"ISNULL",
		};


		public static string[] DataTypes = new string[]
		{
			"INT", "DATETIME", "DECIMAL", "BIT", "NUMERIC", "SMALLDATETIME",
			"DATETIME2", "NVARCHAR"
		};

		private static readonly Dictionary<int, string[]> _sqlFuncArgsCount_SqlFuncNames = new Dictionary<int, string[]>()
		{
			{ 0, SqlFunc0Names },
			{ 1, SqlFunc1Names },
			{ 2, SqlFunc2Names },
		};

		private static readonly Dictionary<string, int> _sqlFuncName_ArgsCount = new Dictionary<string, int>();

		static readonly string[] CompareOperSymbols = new[]
		{
			"!=", "<=", ">=", "<>",
			">", "<", "="
		};

		static readonly string[] CompareOps = new []
		{
			"LIKE", "IN", "IS"
		}.Concat(CompareOperSymbols).ToArray();

		static IEnumerable<string> Escape(IEnumerable<string> patterns)
		{
			return patterns.Select(e => Regex.Escape(e));
		}

		public static string[] Op1 = new []
		{
			"*", "/"
		};

		public static string[] Op2 = new []
		{
			"+", "-"
		};
		
		public static string[] Terms = new []
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
			Escape(CompareOperSymbols)
			.Concat(AllStrings)
			.Concat(new[]
			{
				BatchInstruction,
				SingleLineComment,
				MultiLineComment,
				Hex16Number,
				DecimalNumber,
				IntegerNumber,
				SqlIdent,
				SqlVariable,
				RegexPattern.Ident,
			}).Concat(OtherSymbols)
			.ToArray();

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

		public bool IsIdentWord
		{
			get
			{
				char c = Ch;
				return char.IsLower(c) || char.IsUpper(c) || c == '_';
			}
		}

		public bool IsIdent
		{
			get
			{
				if (_keywords.Contains(Text.ToUpper()))
				{
					return false;
				}
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
				if (_keywords.Contains(Text.ToUpper()))
				{
					return false;
				}
				if (IsFuncName(out _))
				{
					return false;
				}
				if (Text.StartsWith("[") && Text.EndsWith("]"))
				{
					return true;
				}
				if (Text.StartsWith("@"))
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
				throw new System.Exception($"Expect match '{name}' pattern, but got '{Text}'");
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

		public bool IsMatch(string pattern)
		{
			return new Regex("^" + pattern + "$").IsMatch(Text);
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
	}
}
