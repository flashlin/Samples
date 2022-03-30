using Xunit;
using PreviewLibrary;
using FluentAssertions;
using System.Collections.Generic;
using FluentAssertions.Equivalency;
using ExpectedObjects;
using System.Linq;
using Xunit.Abstractions;
using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.IO;
using TestProject.Helpers;

namespace TestProject
{
	public class SetTest : SqlTestBase
	{
		public SetTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void variable_eq_1()
		{
			var sql = "@a = 1";
			var expr = _sqlParser.ParseEqualOpPartial(sql);
			sql.ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void set_variable_eq_xx()
		{
			var sql = "set @id = 1";
			var expr = _sqlParser.ParseSetPartial(sql);
			"SET @id = 1".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void set_variable_eq_case()
		{
			var sql = @"SET @returnValue = CASE WHEN exists(
								select 1 from @A
								where name = 
									cast(@b as nvarchar(3)) + ':' + cast(@c as nvarchar(3)) 
								)
							then 0
							else 1
							end";
			var expr = _sqlParser.ParseSetPartial(sql);
			@"SET @returnValue = CASE
	WHEN exists( SELECT 1 FROM @A WHERE name = CAST( @b AS nvarchar(3) ) + ':' + CAST( @c AS nvarchar(3) ) ) THEN 0
	ELSE 1
END".ToExpectedObject().ShouldEqual(expr.ToString());
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
		public void set_variable_eq_arithmetic()
		{
			var sql = "set @a = 1 + 2";
			var expr = _sqlParser.ParseSetPartial(sql);
			"SET @a = 1 + 2".ShouldEqual(expr);
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

		[Fact]
		public void set_variable_eq_customFunc()
		{
			var sql = "set @a = [dbo].[myFunc]()";
			var expr = _sqlParser.ParseSetPartial(sql);
			"SET @a = [dbo].[myFunc]()".ShouldEqual(expr);
		}


	}
}