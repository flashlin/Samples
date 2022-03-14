using ExpectedObjects;
using PreviewLibrary;
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
            Fields = new List<AssignSetExpr> { 
               new AssignSetExpr
               { 
                  Field = new IdentExpr
                  { 
                     Name = "price"
                  },
                  Value = new AndOrExpr
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
	}
}