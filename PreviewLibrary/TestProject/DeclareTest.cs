using PreviewLibrary;
using Xunit;
using Xunit.Abstractions;
using ExpectedObjects;
using PreviewLibrary.Expressions;

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
         new DeclareVariableExpr
         {
            Name = "@returnValue",
            DataType = new DataTypeExpr
            {
               DataType = "bit"
            }
         }.ToExpectedObject().ShouldEqual(expr);
		}

      [Fact]
      public void declare_variableName_int_eq_default()
      {
         var sql = "DECLARE @returnValue bit = 1";
         var expr = _sqlParser.ParseDeclarePartial(sql);
         sql.ToExpectedObject().ShouldEqual(expr.ToString());
      }

      [Fact]
      public void declare_variableName_decimal_eq_var1_add_var2()
		{
         var sql = "declare @a decimal(19,3) = @b + @c";
         var expr = _sqlParser.ParseDeclarePartial(sql);
         "DECLARE @a decimal(19,3) = @b + @c".ToExpectedObject().ShouldEqual(expr.ToString());
		}
   }
}