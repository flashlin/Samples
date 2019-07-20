using System;
using System.Collections.Generic;
using System.Text;
using ChargeLimitConfig_DesignPattern1.Rules;
using Xunit;

namespace MyTests
{
	public class AmountTest
	{
		[Fact]
		public void Int_UnLimit()
		{
			var a = Amount<int>.From(1);
			var b = Amount<int>.Unlimit;
			Assert.True(a < b);
		}

		[Fact]
		public void UnLimit_Int()
		{
			var a = Amount<int>.Unlimit;
			var b = Amount<int>.From(1);
			Assert.True(a > b);
		}

		[Fact]
		public void UnLimit_UnLimit()
		{
			var a = Amount<int>.Unlimit;
			var b = Amount<int>.Unlimit;
			Assert.True(a == b);
		}

		[Fact]
		public void From0()
		{
			var a = Amount<int>.From(0);
			Assert.Equal(Amount<int>.Unlimit, a);
		}

		[Fact]
		public void FromFloat0()
		{
			var a = Amount<float>.From(0);
			Assert.Equal(Amount<float>.Unlimit, a);
		}

		[Fact]
		public void FromDouble0_dot1()
		{
			var a = Amount<double>.From(0.1d);
			Assert.NotEqual(Amount<double>.Unlimit, a);
		}

		[Fact]
		public void Add()
		{
			var a = Amount<double>.From(1d);
			var b = Amount<double>.From(a.Value + 2d);
			Assert.Equal(Amount<double>.From(3), b);
		}
	}
}
