using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using Microsoft.EntityFrameworkCore;
using Xunit;

namespace LinqJoinTutor
{
	public class LinqSqlTest
	{
		[Fact]
		public void group()
		{
			using var db = new MyDbContext();
			var q1 = from c in db.Customers
						group c by c.Name
				into g1
						select new
						{
							Key = g1.Key,
							Prices = g1.ToList()
						};
			var sql = q1.ToQueryString();
			var data = q1.ToArray();
		}

		[Fact]
		public void JoinOuter1()
		{
			using var mydb = new MyDbContext();
			var q1 = from input in mydb.Customers
				join db in mydb.Houses on input.Id equals db.CustomerId into ps
				from dbb in ps.DefaultIfEmpty(new House { Address = "" })
				select new
				{
					inputProductGuid = input.Id,
					inputProductName = input.Name,
					dbProductGuid = dbb.Address,
				};
			var sql = q1.ToQueryString();
		}
		[Fact]
		public void JoinOuter2()
		{
			using var mydb = new MyDbContext();
			var q1 = from input in mydb.Customers
						join db in mydb.Houses on input.Id equals db.CustomerId
						select new
						{
							inputProductGuid = input.Id,
							inputProductName = input.Name,
							dbProductGuid = db.Address
						};
			var sql = q1.ToQueryString();
		}
	}

	public class JoinTest
	{
		[Fact]
		public void InnerJoin()
		{
			var query = from tb1 in GetProduct()
							join tb2 in GetCategories() on tb1.CategoryId equals tb2.Id
							select new
							{
								CategoryName = tb2.Name,
								ProductName = tb1.Name
							};

			Assert.Equal(
				new[]
				{
					new
					{
						CategoryName = "新竹",
						ProductName = "鳳梨"
					},
					new
					{
						CategoryName = "台中",
						ProductName = "香蕉"
					}
				},
				query.ToArray());
		}


		[Fact]
		public void LeftJoin()
		{
			var query = from tb1 in GetProduct()
							join tb2 in GetCategories() on tb1.CategoryId equals tb2.Id into cp
							from tb2 in cp.DefaultIfEmpty()
							select new
							{
								CategoryName = tb2?.Name,
								ProductName = tb1.Name
							};

			Assert.Equal(
				new[]
				{
					new
					{
						CategoryName = (string)null,
						ProductName = "葡萄"
					},
					new
					{
						CategoryName = "新竹",
						ProductName = "鳳梨"
					},
					new
					{
						CategoryName = "台中",
						ProductName = "香蕉"
					},
				},
				query.ToArray());
		}

		[Fact]
		public void RightJoin()
		{
			var query = from tb2 in GetCategories()
							join tb1 in GetProduct() on tb2.Id equals tb1.CategoryId into cp
							from tb1 in cp.DefaultIfEmpty()
							select new
							{
								CategoryName = tb2.Name,
								ProductName = tb1?.Name
							};

			Assert.Equal(
				new[]
				{
					new
					{
						CategoryName = "新竹",
						ProductName = "鳳梨"
					},
					new
					{
						CategoryName = "台中",
						ProductName = "香蕉"
					},
					new
					{
						CategoryName = "高雄",
						ProductName = (string)null
					},
				},
				query.ToArray());
		}

		[Fact]
		public void LeftJoin1()
		{
			var query = from tb1 in GetProduct()
							join tb2 in GetCategories() on tb1.CategoryId equals tb2.Id into cp
							from tb2 in cp.DefaultIfEmpty()
							where tb2 == null
							select new
							{
								CategoryName = (string)null,
								ProductName = tb1.Name
							};

			Assert.Equal(
				new[]
				{
					new
					{
						CategoryName = (string)null,
						ProductName = "葡萄"
					},
				},
				query.ToArray());
		}

		[Fact]
		public void Test5()
		{
			var query = from tb2 in GetCategories()
							join tb1 in GetProduct() on tb2.Id equals tb1.CategoryId into cp
							from tb1 in cp.DefaultIfEmpty()
							where tb1 == null
							select new
							{
								CategoryName = tb2.Name,
								ProductName = (string)null
							};

			Assert.Equal(
				new[]
				{
					new
					{
						CategoryName = "高雄",
						ProductName = (string)null
					},
				},
				query.ToArray());
		}







		[Fact]
		public void Test6()
		{
			var leftJoinQuery = from tb1 in GetProduct()
									  join tb2 in GetCategories() on tb1.CategoryId equals tb2.Id into cp
									  from tb2 in cp.DefaultIfEmpty()
									  where tb2 == null
									  select new
									  {
										  CategoryName = (string)null,
										  ProductName = tb1.Name
									  };

			var rightJoinQuery = from tb2 in GetCategories()
										join tb1 in GetProduct() on tb2.Id equals tb1.CategoryId into cp
										from tb1 in cp.DefaultIfEmpty()
										where tb1 == null
										select new
										{
											CategoryName = tb2.Name,
											ProductName = (string)null
										};

			var result = leftJoinQuery.Union(rightJoinQuery)
				.ToArray();

			Assert.Equal(
				new[]
				{
					new
					{
						CategoryName = (string)null,
						ProductName = "葡萄"
					},
					new
					{
						CategoryName = "高雄",
						ProductName = (string)null
					},
				},
				result);
		}

		[Fact]
		public void FullOuterJoin()
		{
			var leftOuterJoin = from tb1 in GetProduct()
									  join tb2 in GetCategories() on tb1.CategoryId equals tb2.Id into cp
									  from tb2 in cp.DefaultIfEmpty()
									  select new
									  {
										  CategoryName = tb2?.Name,
										  ProductName = tb1.Name
									  };

			var rightOuterJoin = from tb2 in GetCategories()
										join tb1 in GetProduct() on tb2.Id equals tb1.CategoryId into cp
										from tb1 in cp.DefaultIfEmpty()
										select new
										{
											CategoryName = tb2.Name,
											ProductName = tb1?.Name
										};

			var result = leftOuterJoin.Union(rightOuterJoin);
			Assert.Equal(
				new[]
				{
					new
					{
						CategoryName = (string)null,
						ProductName = "葡萄"
					},
					new
					{
						CategoryName = "新竹",
						ProductName = "鳳梨"
					},
					new
					{
						CategoryName = "台中",
						ProductName = "香蕉"
					},
					new
					{
						CategoryName = "高雄",
						ProductName = (string)null
					},
				},
				result);
		}

		private static IEnumerable<Category> GetCategories()
		{
			yield return new Category()
			{
				Id = 2,
				Name = "新竹"
			};
			yield return new Category()
			{
				Id = 3,
				Name = "台中"
			};
			yield return new Category()
			{
				Id = 4,
				Name = "高雄"
			};
		}

		private static IEnumerable<Product> GetProduct()
		{
			yield return new Product()
			{
				CategoryId = 1,
				Name = "葡萄",
				Price = 150
			};
			yield return new Product()
			{
				CategoryId = 2,
				Name = "鳳梨",
				Price = 300
			};
			yield return new Product()
			{
				CategoryId = 3,
				Name = "香蕉",
				Price = 200
			};
		}
	}

	public class Category
	{
		public int Id { get; set; }
		public string Name { get; set; }
	}

	public class Product
	{
		public string Name { get; set; }
		public int Price { get; set; }
		public int CategoryId { get; set; }
	}
}
