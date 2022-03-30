using ExpectedObjects;
using PreviewLibrary.Exceptions;
using PreviewLibrary;
using TestProject.Helpers;
using System.Collections.Generic;
using System.Linq;

namespace TestProject.Helpers
{
	public static class TestExtension
	{
		public static void ShouldEqual(this string expected, SqlExpr sqlExpr)
		{
			var sqlExprCode = $"{sqlExpr}";
			expected.MergeToCode().TrimCode().ToExpectedObject().ShouldEqual(sqlExprCode.MergeToCode().TrimCode());
		}

		public static void ShouldEqual(this string expected, IEnumerable<SqlExpr> sqlExprList)
		{
			var exprsCode = string.Join("\r\n", sqlExprList.Select(x => $"{x}"));
			expected.TrimCode().ToExpectedObject().ShouldEqual(exprsCode.TrimCode());
		}
	}
}