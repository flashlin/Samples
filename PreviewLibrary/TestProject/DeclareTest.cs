using PreviewLibrary;
using Xunit;
using Xunit.Abstractions;
using ExpectedObjects;
using PreviewLibrary.Expressions;
using TestProject.Helpers;

namespace TestProject
{
	public class DeclareTest : SqlTestBase
	{
		public DeclareTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

      [Fact]
      public void declare_variableName_int()
		{
         var sql = "declare @returnValue bit";
         
         var expr = Parse(sql);

         "DECLARE @returnValue bit".ShouldEqual(expr);
		}

      [Fact]
      public void declare_two_variable()
		{
         var sql = "declare @a decimal(19, 6) = 1, @b decimal(19, 6) = 2";
         
         var expr = _sqlParser.ParseDeclarePartial(sql);

         @"DECLARE @a decimal(19,6) = 1
DECLARE @b decimal(19,6) = 2".ShouldEqual(expr);
		}

      [Fact]
      public void declare_variableName_int_eq_default()
      {
         var sql = "DECLARE @returnValue bit = 1";
         var expr = _sqlParser.ParseDeclarePartial(sql);
         sql.ShouldEqual(expr);
      }

      [Fact]
      public void declare_variableName_decimal_eq_var1_add_var2()
		{
         var sql = "declare @a decimal(19,3) = @b + @c";
         var expr = _sqlParser.ParseDeclarePartial(sql);
         "DECLARE @a decimal(19,3) = @b + @c".ShouldEqual(expr);
		}

      [Fact]
      public void declare_variableName_as_datatype()
		{
         var sql = "declare @a as int";
         
         var expr = _sqlParser.ParseDeclarePartial(sql);

         "DECLARE @a int".ShouldEqual(expr);
		}


   }
}