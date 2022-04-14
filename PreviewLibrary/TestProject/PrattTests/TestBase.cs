using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.TSql;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit.Abstractions;
using PreviewLibrary.Extensions;
using FluentAssertions;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.IO;
using PreviewLibrary.Pratt.Core.Expressions;

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
			_expr.ToString().MergeToCode().Should().Be(expectCode.MergeToCode());
		}

		protected IEnumerable<string> ReadSqlFiles(string folder)
		{
			var sqlFiles = Directory.EnumerateFiles(folder, "*.sql");
			var subDirs = Directory.EnumerateDirectories(folder);
			return sqlFiles.Concat(subDirs.SelectMany(x => ReadSqlFiles(x)));
		}
	}
}
