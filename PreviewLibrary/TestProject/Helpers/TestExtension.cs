using ExpectedObjects;
using PreviewLibrary.Exceptions;
using PreviewLibrary;
using TestProject.Helpers;

namespace TestProject.Helpers
{
	public static class TestExtension
	{
		public static void ShouldEqual(this string expected, SqlExpr sqlExpr)
		{
			expected.TrimCode().ToExpectedObject().ShouldEqual(sqlExpr.ToString().TrimCode());
		}
	}
}