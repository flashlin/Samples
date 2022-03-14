using PreviewLibrary.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using T1.Standard.Extensions;

namespace PreviewLibrary
{
	public abstract class TokenizerBase : LineChInfo
	{
		private readonly string _pattern;
		//private IEnumerator<Group> _tokens;
		private ITwoWayEnumerator<Group> _tokens;

		public TokenizerBase(IEnumerable<string> regexPatternList)
		{
			_pattern = @"\s*("
				+ string.Join("|", regexPatternList)
				+ @")\s*";
		}

		public int CurrentIndex
		{
			get
			{
				return _tokens.CurrentIndex;
			}
		}

		public Group Curr
		{
			get
			{
				return _tokens.Current;
			}
		}

		public string Text
		{
			get
			{
				if (Curr == null)
				{
					return string.Empty;
				}
				return Curr.Value ?? string.Empty;
			}
		}

		public char Ch
		{
			get
			{
				return string.IsNullOrEmpty(Text) ? ' ' : Text[0];
			}
		}

		public bool IsInteger
		{
			get
			{
				return Int32.TryParse(Text, out var _);
			}
		}

		public bool IsNumber
		{
			get
			{
				return char.IsNumber(Ch);
			}
		}

		public bool IsDouble
		{
			get
			{
				return IsNumber && Text.Contains('.');
			}
		}

		public bool IsString
		{
			get
			{
				return Ch == '"';
			}
		}

		public string CurrAndNext()
		{
			string s = Text;
			if (!Move()) Abort("data expected"); return s;
		}

		public string CurrOptNext()
		{
			string s = Text;
			Move();
			return s;
		}

		public string CurrOpAndNext(params string[] ops)
		{
			string s = ops.Contains(Text) ? Text : null;
			if (s != null && !Move()) Abort("data expected");
			return s;
		}

		public string Next()
		{
			var s = Text;
			Move();
			return s;
		}

		public void PredicateParse(string s)
		{
			_tokens = Regex.Matches(s, _pattern, RegexOptions.Compiled).Cast<Match>()
						 .Select(m => m.Groups[1])
						 .GetTwoWayEnumerator();
			Move();
		}

		public bool Move()
		{
			var success = false;
			do
			{
				success = _tokens.MoveNext();
			} while (success && Text == "\r\n");
			return success;
		}

		public bool MovePrevious()
		{
			var success = false;
			do
			{
				success = _tokens.MovePrevious();
			} while (success && Text == "\r\n");
			return success;
		}

		public bool MoveTo(int index)
		{
			return _tokens.MoveTo(index);
		}

		protected void Abort(string msg)
		{
			throw new ArgumentException("Error: " + (msg ?? "unknown error"));
		}

		public bool Try(string text)
		{
			return Try(text, out _);
		}

		public bool Try(bool success, out string curr)
		{
			curr = null;
			if (success)
			{
				curr = Text;
				Move();
			}
			return success;
		}

		public bool Try(string text, out string output)
		{
			output = null;
			var success = (Text == text);
			if (success)
			{
				output = Text;
				Move();
			}
			return success;
		}

		public bool IgnoreCase(string text)
		{
			return string.Equals(Text, text, StringComparison.OrdinalIgnoreCase);
		}

		public bool IgnoreCaseAny(params string[] strs)
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

		public bool TryIgnoreCase(string text)
		{
			return TryIgnoreCase(text, out var _);
		}

		public bool TryIgnoreCase(string text, out string output)
		{
			output = null;
			var success = string.Equals(Text, text, StringComparison.OrdinalIgnoreCase);
			if (success)
			{
				output = Text;
				Move();
			}
			return success;
		}


		public bool TryIgnoreCase(string[] strs, out string output)
		{
			for (var i = 0; i < strs.Length; i++)
			{
				if (TryIgnoreCase(strs[i], out output))
				{
					return true;
				}
			}
			output = null;
			return false;
		}

		public bool TryInteger(out int output)
		{
			var success = Int32.TryParse(Text, out output);
			if (success)
			{
				Move();
			}
			return success;
		}

		public LineChInfo GetLineCh(string content)
		{
			if (Curr == null)
			{
				return new LineChInfo
				{
					LineNumber = 0,
					ChNumber = 0,
					PrevLines = new string[0],
					Line = String.Empty,
				};
			}

			var previewContent = content.Substring(0, Curr.Index);
			var lines = previewContent.Split("\r\n");
			var line = lines[lines.Length - 1];
			var prevLines = lines.SkipLast(1).TakeLast(3).ToArray();
			return new LineChInfo
			{
				LineNumber = lines.Length,
				ChNumber = line.Length + 1,
				PrevLines = prevLines,
				Line = line
			};
		}
	}
}
