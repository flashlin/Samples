using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit.Abstractions;
using FluentAssertions;
using System.IO;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;
using Xunit;

namespace TestProject.PrattTests
{
	public abstract class TestBase
	{
		protected readonly ITestOutputHelper _outputHelper;
		protected TSqlParser _parser;
		private List<IExpression> _exprList;
		protected SqlCodeExpr _expr;
		private TSqlScanner _scanner;
		private List<TextSpan> _tokenList;
		private TextSpan _token;

		public TestBase(ITestOutputHelper outputHelper)
		{
			_outputHelper = outputHelper;
		}

		public void Parse(string text)
		{
			var scanner = new TSqlScanner(text);
			_parser = new TSqlParser(scanner);
			_expr = _parser.ParseExpression();
		}

		public void ParseAll(string text)
		{
			var scanner = new TSqlScanner(text);
			_parser = new TSqlParser(scanner);
			_exprList = _parser.ParseProgram().ToList();
		}

		protected void Scan(string text)
		{
			_scanner = new TSqlScanner(text);
			_token = _scanner.Consume();
		}

		protected void ScanAll(string text)
		{
			_scanner = new TSqlScanner(text);
			_tokenList = new List<TextSpan>();
			do
			{
				_token = _scanner.Consume();
				if (_token.IsEmpty)
				{
					break;
				}
				_tokenList.Add(_token);
			} while (true);
		}

		protected void ThenTokenShouldBe(string expectToken)
		{
			var tokenStr = _scanner.GetSpanString(_token);
			tokenStr.Should().Be(expectToken);
		}

		protected void ThenTokenListShouldBe(params string[] expectTokenList)
		{
			foreach (var expectToken in expectTokenList.Select((val, idx) => new { val, idx }))
			{
				var tokenStr = _scanner.GetSpanString(_tokenList[expectToken.idx]);
				tokenStr.Should().Be(expectToken.val);
			}
		}

		protected void ThenExprShouldBe(string expectCode)
		{
			var exprCode = _expr.ToString();
			//exprCode.MergeToCode().Should().Be(expectCode.MergeToCode());
			Assert.Equal(expectCode.MergeToCode(), exprCode.MergeToCode());
		}

		protected IEnumerable<string> ReadSqlFiles(string folder)
		{
			var sqlFiles = Directory.EnumerateFiles(folder, "*.sql");
			var subDirs = Directory.EnumerateDirectories(folder);
			return sqlFiles.Concat(subDirs.SelectMany(x => ReadSqlFiles(x)));
		}

		protected string GetCompareCodeHelpMessage(string text, string expect)
		{
			var textLines = text.ToLines();
			var expectLines = expect.ToLines();

			var sb = new StringBuilder();
			var isEqual = true;
			var minLines = Math.Min(textLines.Count, expectLines.Count);
			for (var line = 0; line < minLines; line++)
			{
				var s1 = textLines[line];
				var s2 = expectLines[line];
				if (s1 != s2)
				{
					isEqual = false;
					var help = GetCompareStringHelpMessage(s1, s2);
					sb.AppendLine(help);
				}
			}

			if (isEqual)
			{
				return String.Empty;
			}

			var sb2 = new StringBuilder();
			sb2.AppendLine("=== BEGIN ===");
			sb2.AppendLine(sb.ToString());
			sb2.AppendLine("=== END ===");
			return sb2.ToString();
		}

		protected string GetCompareStringHelpMessage(string text, string expect)
		{
			var diffIndex = DiffersAtIndex(text, expect);
			var sb = new StringBuilder();
			if (diffIndex > 0)
			{
				sb.Append(new String(' ', diffIndex));
			}
			sb.AppendLine("v");
			sb.AppendLine(text);
			if (diffIndex > 0)
			{
				sb.Append(new String(' ', diffIndex));
			}
			sb.AppendLine("^");
			sb.AppendLine(expect);
			return sb.ToString();
		}

		static int DiffersAtIndex(string s1, string s2)
		{
			int index = 0;
			int min = Math.Min(s1.Length, s2.Length);
			while (index < min && s1[index] == s2[index])
				index++;
			return (index == min && s1.Length == s2.Length) ? -1 : index;
		}
	}
}
