using ExpectedObjects;
using PreviewLibrary;
using Xunit;
using Xunit.Abstractions;

namespace TestProject
{
	public class ArithmeticTest : SqlTestBase
	{
		public ArithmeticTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void add_mul()
		{
			var sql = "1 + 2 * 3";
			var expr = new SqlParser().ParseArithmeticExpression(sql);
			new AndOrExpr
         {
            Left = new IntegerExpr
            {
               Value = 1
            },
            Oper = "+",
            Right = new AndOrExpr
            {
               Left = new IntegerExpr
               {
                  Value = 2
               },
               Oper = "*",
               Right = new IntegerExpr
               {
                  Value = 3
               }
            }
         }.ToExpectedObject().ShouldEqual(expr);
		}

      [Fact]
      public void add_mul_add()
      {
         var sql = "1 + 2 * 3 + 4";
         var expr = new SqlParser().ParseArithmeticExpression(sql);
         new AndOrExpr
         {
            Left = new IntegerExpr
            {
               Value = 1
            },
            Oper = "+",
            Right = new AndOrExpr
            {
               Left = new AndOrExpr
               {
                  Left = new IntegerExpr
                  {
                     Value = 2
                  },
                  Oper = "*",
                  Right = new IntegerExpr
                  {
                     Value = 3
                  }
               },
               Oper = "+",
               Right = new IntegerExpr
               {
                  Value = 4
               }
            }
         }.ToExpectedObject().ShouldEqual(expr);
      }
   }
}