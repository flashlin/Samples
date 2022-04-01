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
		public void exec_func_arg1_arg2()
		{
			var sql = @"exec sys.sp_addextendedproperty @name = N'MS_Description', @value = N'Confidential.';";
			var expr = Parse(sql);
			new ExecuteExpr
			{
				ExecName = "exec",
				Method = new IdentExpr
				{
					ObjectId = "sys",
					Name = "sp_addextendedproperty"
				},
				Arguments = new SqlExpr[]
				{
					new SpParameterExpr
					{
						Name = "@name",
						Value = new StringExpr
						{
							Text = "N'MS_Description'"
						}
					},
					new SpParameterExpr
					{
						Name = "@value",
						Value = new StringExpr
						{
							Text = "N'Confidential.'"
						}
					}
				}
			}.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void exec_func_arg_out()
		{
			var sql = @"exec [my_func] @a out";
			var expr = _sqlParser.ParseExecPartial(sql);
			"EXEC [my_func] @a OUT".ShouldEqual(expr);
		}



		[Fact]
		public void execute_func_arg1()
		{
			var sql = @"execute sys.sp_addextendedproperty @name = N'MS_Description';";
			var expr = Parse(sql);
			new ExecuteExpr
			{
				ExecName = "execute",
				Method = new IdentExpr
				{
					ObjectId = "sys",
					Name = "sp_addextendedproperty"
				},
				Arguments = new SqlExpr[]
				{
					new SpParameterExpr
					{
						Name = "@name",
						Value = new StringExpr
						{
							Text = "N'MS_Description'"
						}
					},
				}
			}.ToExpectedObject().ShouldEqual(expr);
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

		[Fact]
		public void exec_variable_eq_funcname()
		{
			var sql = @"exec @a = customFunc";
			var expr = _sqlParser.ParseExecPartial(sql);
			"EXEC @a = customFunc".ShouldEqual(expr);
		}
	}
}