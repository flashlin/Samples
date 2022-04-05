using PreviewLibrary.PrattParsers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit.Abstractions;

namespace TestProject.ScannerTests
{
	public abstract class ScanTestBase
	{
		protected readonly ITestOutputHelper _outputHelper;
		protected IScanner _scanner;

		public ScanTestBase(ITestOutputHelper outputHelper)
		{
			_outputHelper = outputHelper;
		}

		protected void Scan(string text)
		{
			_scanner = new StringScanner(text);
		}
	}
}
