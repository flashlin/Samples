using System;
using System.Linq;
using Xunit.Sdk;
using System.Collections.Generic;
using Xunit.Abstractions;

namespace SqlLocalDataTests
{
	public class AlphabeticalOrderer : ITestCaseOrderer
	{
		public IEnumerable<TTestCase> OrderTestCases<TTestCase>(IEnumerable<TTestCase> testCases)
			where TTestCase : ITestCase
		{
			var result = testCases.ToList();
			result.Sort((x, y) => StringComparer.OrdinalIgnoreCase.Compare(x.TestMethod.Method.Name, y.TestMethod.Method.Name));
			return result;
		}
	}
}