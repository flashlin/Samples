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
      public void declare_varname_int()
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
	}
}