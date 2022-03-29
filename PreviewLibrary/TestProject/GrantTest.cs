using Xunit;
using PreviewLibrary;
using ExpectedObjects;
using Xunit.Abstractions;
using TestProject.Helpers;

namespace TestProject
{
	public class GrantTest : SqlTestBase
	{
		public GrantTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void grant_execute_on_object()
		{
			var sql = "GRANT EXECUTE ON OBJECT::[dbo].[aaa] TO [RoleName] AS[dbo]";
			var expr = Parse(sql);
			"GRANT EXECUTE ON OBJECT::[dbo].[aaa] TO [RoleName] AS [dbo]".ShouldEqual(expr);
		}

		[Fact]
		public void grant_execute_on_dbo_object_to_role()
		{
			var sql = "grant execute on [dbo].[fn_name] TO RolePlayer";
			var expr = _sqlParser.ParseGrantPartial(sql);
			"GRANT EXECUTE ON [dbo].[fn_name] TO RolePlayer".ShouldEqual(expr);
		}

		[Fact]
		public void grant_execute_on_objectId()
		{
			var sql = "GRANT EXEC ON [dbo].[a] TO RoleUser";
			var expr = _sqlParser.ParseGrantPartial(sql);
			"GRANT EXEC ON [dbo].[a] TO RoleUser".ShouldEqual(expr);
		}


	}
}