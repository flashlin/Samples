using ExpectedObjects;
using PreviewLibrary.Exceptions;
using TestProject.Helpers;

namespace TestProject.Helpers
{
	public static class TestExtension
	{
		public static void ShouldEqual(this string expected, SqlExpr sqlExpr)
		{
			expected.ToExpectedObject().ShouldEqual(sqlExpr.ToString());
		}
	}
}