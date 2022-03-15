using Xunit;
using PreviewLibrary;
using ExpectedObjects;
using Xunit.Abstractions;

namespace TestProject
{
	public class GrantTest : SqlTestBase
	{
		public GrantTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void grant_execute_on()
		{
			var sql = "GRANT EXECUTE ON OBJECT::[dbo].[Leo_Account_UpdateTracDelayStatus_18.08] TO [RoleLeo] AS[dbo]";
			var expr = Parse(sql);
			new GrantExecuteOnExpr
			{
				ToRoleId = new IdentExpr
				{
					Name = "[RoleLeo]"
				},
				AsDbo = new IdentExpr
				{
					Name = "[dbo]"
				},
				OnObjectId = new ObjectIdExpr
				{
					Name = new IdentExpr
					{
						Name = "[Leo_Account_UpdateTracDelayStatus_18.08]",
						ObjectId = "[dbo]"
					}
				}
			}.ToExpectedObject().ShouldEqual(expr);
		}
	}
}