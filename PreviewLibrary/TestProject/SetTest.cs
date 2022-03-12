using Xunit;
using PreviewLibrary;
using FluentAssertions;
using System.Collections.Generic;
using FluentAssertions.Equivalency;
using ExpectedObjects;
using System.Linq;
using Xunit.Abstractions;

namespace TestProject
{

	public class SetTest : SqlTestBase
	{
		public SetTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void set_xxx_off()
		{
			var sql = "SET NUMERIC_ROUNDABORT OFF";
			var expr = Parse(sql);
			new SetOptionsExpr
			{
				Options = new List<string>
				{
					"NUMERIC_ROUNDABORT"
				},
				Toggle = "OFF"
			}.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void SetManyOptionsOn()
		{
			var sql = "SET ANSI_NULLS, ANSI_PADDING ON;";
			var expr = Parse(sql);
			var expected = new SetOptionsExpr
			{
				Options = new List<string>
				{
					"ANSI_NULLS",
					"ANSI_PADDING"
				},
				Toggle = "ON"
			};
			expected.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void SetVariable()
		{
			var sql = ":setvar DatabaseName \"AccountDB\"";
			var expr = Parse(sql);
			new SetBatchVariableExpr
			{
				Name = "DatabaseName",
				Value = "\"AccountDB\"",
			}.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void On_Error_Exit()
		{
			var sql = ":on error exit";
			var expr = Parse(sql);
			new OnConditionThenExpr
			{
				Condition = "error",
				ActionName = "exit"
			}.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void Print()
		{
			var sql = "PRINT N'SQLCMD mode must be enabled to successfully execute this script.';";
			var expr = Parse(sql);
			new PrintExpr
			{
				Content = new StringExpr
				{
					Text = "N'SQLCMD mode must be enabled to successfully execute this script.'"
				}
			}.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void use()
		{
			var sql = "USE [$(DatabaseName)];";
			var expr = Parse(sql);
			new UseExpr
			{
				ObjectId = new IdentExpr
				{
					Name = "[$(DatabaseName)]"
				}
			}.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void select_not_exists()
		{
			var sql = "select not exists(1)";
			var expr = Parse(sql);
			new SelectExpr
			{
				Fields = new List<SqlExpr>
				{
					new NotExpr
					{
						Right = new SqlFuncExpr
						{
							Name = "exists",
							Arguments = new SqlExpr[]
							{
								new IntegerExpr
								{
									Value = 1
								}
							}
						}
					}
				}
			}
			.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void select_1_from_table_where_name_eq_func0_and_func1_eq_string()
		{
			var sql = @"select 1 from sys.databases where name = DB_NAME() and SUSER_SNAME(owner_sid) = 'sa'";
			var expr = Parse(sql);
			new SelectExpr
			{
				Fields = new List<SqlExpr>
				{
					new IntegerExpr { Value = 1 }
				},
				From = new TableExpr
				{
					Name = new IdentExpr
					{
						ObjectId = "sys",
						Name = "databases"
					}
				},
				WhereExpr = new AndOrExpr
				{
					Left = new CompareExpr
					{
						Left = new IdentExpr { Name = "name" },
						Oper = "=",
						Right = new SqlFuncExpr { Name = "DB_NAME", Arguments = new SqlExpr[0] }
					},
					Oper = "and",
					Right = new CompareExpr
					{
						Left = new SqlFuncExpr
						{
							Name = "SUSER_SNAME",
							Arguments = new SqlExpr[]
							{
								new IdentExpr { Name = "owner_sid" }
							}
						},
						Oper = "=",
						Right = new StringExpr
						{
							Text = "'sa'"
						}
					}
				}
			}
			.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void select_1_from_table_where_name_eq_func0()
		{
			var sql = @"select 1 from sys.databases where name = DB_NAME()";
			var expr = Parse(sql);
			new SelectExpr
			{
				Fields = new List<SqlExpr>
				{
					new IntegerExpr { Value = 1 }
				},
				From = new TableExpr
				{
					Name = new IdentExpr
					{
						ObjectId = "sys",
						Name = "databases"
					}
				},
				WhereExpr = new CompareExpr
				{
					Left = new IdentExpr { Name = "name" },
					Oper = "=",
					Right = new SqlFuncExpr { Name = "DB_NAME", Arguments = new SqlExpr[0] }
				},
			}
			.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void if_not_func1_selectExpr_begin_selectExpr_end()
		{
			var sql = @"if not exists(
    select 1 from sys.databases
)
begin
	select 1
end";
			var expr = Parse(sql);
			new IfExpr
			{
				Condition = new NotExpr
				{
					Right = new SqlFuncExpr
					{
						Name = "exists",
						Arguments = new SqlExpr[]
						{
							new SelectExpr
							{
								Fields = new List<SqlExpr>{ new IntegerExpr { Value = 1 } },
								From = new TableExpr
								{
									Name = new IdentExpr
									{
										ObjectId = "sys",
										Name = "databases"
									}
								}
							}
						}
					}
				},
				Body = new List<SqlExpr>
				{
					new SelectExpr
					{
						Fields = new List<SqlExpr> { new IntegerExpr { Value = 1 } },
					}
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
		public void set_identity_insert_objectId_on()
		{
			var sql = "SET IDENTITY_INSERT [dbo].[customer] ON";
			var expr = Parse(sql);
			new SetPermissionExpr
			{
				Permission = "IDENTITY_INSERT",
				ToObjectId = new IdentExpr
				{
					Name = "[customer]",
					ObjectId = "[dbo]"
				},
				Toggle = true
			}.ToExpectedObject().ShouldEqual(expr);
		}
	}
}