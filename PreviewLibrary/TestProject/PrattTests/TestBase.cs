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

namespace TestProject.PrattTests
{
	public abstract class TestBase
	{
		protected readonly ITestOutputHelper _outputHelper;
		protected IParser<SqlCodeDom> _parser;
		private SqlCodeDom _expr;

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

		protected void ThenExprShouldBe(string expectCode)
		{
			_expr.ToString().MergeToCode().Should().Be(expectCode.MergeToCode());
		}
	}
}
