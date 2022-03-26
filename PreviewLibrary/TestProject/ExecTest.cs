using Xunit;
using PreviewLibrary;
using ExpectedObjects;
using Xunit.Abstractions;
using PreviewLibrary.Exceptions;
using TestProject.Helpers;

namespace TestProject
{
	public class ExecTest : SqlTestBase
	{
		public ExecTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void exec_objectId_arg1()
		{
			var sql = @"exec sys.sp_changedbowner 'sa'";
			var expr = Parse(sql);
			new ExecuteExpr
			{
				ExecName = "exec",
				Method = new IdentExpr
				{
					ObjectId = "sys",
					Name = "sp_changedbowner"
				},
				Arguments = new SqlExpr[]
				{
					new StringExpr
					{
						Text = "'sa'"
					}
				}
			}.ToExpectedObject().ShouldEqual(expr);
		}
	}
}