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

		[Fact]
		public void grant_execute_on_dbobject_to_role()
		{
			var sql = "grant execute on [dbo].[fn_name] TO RolePlayer";
			var expr = _sqlParser.ParseGrantPartial(sql);
			"GRANT EXECUTE ON [dbo].[fn_name] TO RolePlayer".ToExpectedObject().ShouldEqual(expr.ToString());
		}
	}
}