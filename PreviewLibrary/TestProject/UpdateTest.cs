using ExpectedObjects;
using PreviewLibrary;
using PreviewLibrary.Exceptions;
using System.Collections.Generic;
using Xunit;
using Xunit.Abstractions;

namespace TestProject
{
	public class UpdateTest : SqlTestBase
	{
		public UpdateTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void update_table_set_field_eq_field_add_1()
		{
			var sql = "Update customer set price = rate + 1";
			var expr = Parse(sql);
			new UpdateExpr
			{
				Fields = new List<SqlExpr>
				{
					new AssignSetExpr
					{
						Field = new IdentExpr
						{
							Name = "price"
						},
						Value = new OperandExpr
						{
							Left = new IdentExpr
							{
								Name = "rate"
							},
							Oper = "+",
							Right = new IntegerExpr
							{
								Value = 1
							}
						}
					}
				}
			}.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void update_table_set_field_eq_variable()
		{
			var sql = "UPDATE [dbo].[TracDelay] SET [Status] = @Status WHERE[Id] = @Id";
			var expr = Parse(sql);
			new UpdateExpr
			{
				Fields = new List<SqlExpr>
				{
					new AssignSetExpr
					{
						Field = new IdentExpr
						{
							 Name = "[Status]"
						},
						Value = new IdentExpr
						{
							 Name = "@Status"
						}
					}
				},
				WhereExpr = new CompareExpr
				{
					Left = new IdentExpr
					{
						Name = "[Id]"
					},
					Oper = "=",
					Right = new IdentExpr
					{
						Name = "@Id"
					}
				}
			}.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void update_set_case_when()
		{
			var sql = @"UPDATE [dbo].[TracDelay] 
SET [ExchangeRate] = CASE WHEN @ExchangeRate = -1 THEN[ExchangeRate] ELSE @ExchangeRate END";
			var expr = Parse(sql);
			new UpdateExpr
			{
				Fields = new List<SqlExpr> { 
					new AssignSetExpr
					{
						Field = new IdentExpr
						{
							Name = "[ExchangeRate]"
						},
						Value = new CaseExpr
						{
							WhenList = new List<WhenThenExpr> { 
								new WhenThenExpr
								{
									When = new CompareExpr
									{
										Left = new IdentExpr
										{
											Name = "@ExchangeRate"
										},
										Oper = "=",
										Right = new IntegerExpr
										{
											Value = -1
										}
									},
									Then = new IdentExpr
									{
										Name = "[ExchangeRate]"
									}
								}
							},
							Else = new IdentExpr
							{
								Name = "@ExchangeRate"
							}
						}
					}
				}
			}.ToExpectedObject().ShouldEqual(expr);
		}
	}
}