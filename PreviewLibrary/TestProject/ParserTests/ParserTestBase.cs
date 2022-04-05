using PreviewLibrary.PrattParsers;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit.Abstractions;

namespace TestProject.ParserTests
{
	public abstract class ParserTestBase
	{
		protected readonly ITestOutputHelper _outputHelper;
		protected IParser _parser;
		protected SqlDom _expr;

		public ParserTestBase(ITestOutputHelper outputHelper)
		{
			_outputHelper = outputHelper;
		}

		protected void Parse(string text)
		{
			var scanner = new StringScanner(text);
			_parser = new SqlParser(scanner);
			_expr = _parser.ParseProgram();
		}
	}
}
