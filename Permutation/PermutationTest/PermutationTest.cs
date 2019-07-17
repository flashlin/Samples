using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Permutation;

namespace PermutationTest
{
	[TestClass]
	public class PermutationTest
	{
		[TestMethod]
		public void One_Two_Three()
		{
			var t = new Permutate();
			var list = t.Compute("123", 0);

			Assert.AreEqual(6, list.Count);
		}


		[TestMethod]
		public void Method2()
		{
			var t = new Permutate();
			var list = t.Compute2("123", 0);

			Assert.AreEqual(6, list.Count);
		}

	}
}
