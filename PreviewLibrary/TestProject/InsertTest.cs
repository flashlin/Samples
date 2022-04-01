using Xunit;
using PreviewLibrary;
using ExpectedObjects;
using Xunit.Abstractions;
using System.Collections.Generic;
using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using TestProject.Helpers;

namespace TestProject
{
	public class InsertTest : SqlTestBase
	{
		public InsertTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void insert_table_fields_values_value1()
		{
			var sql = @"INSERT [dbo].[customer] ([id], [name], [lastname], [birth], [price]) VALUES 
				(267467, N'', NULL, CAST(0x0000A5E5006236FB AS DateTime), 1)";

			var expr = Parse(sql);

			new InsertExpr
			{
				Table = new IdentExpr
				{
					ObjectId = "[dbo]",
					Name = "[customer]",
				},
				Fields = new SqlExprList
				{
					Items = new List<SqlExpr>
					{
						new IdentExpr { Name = "[id]" },
						new IdentExpr { Name = "[name]" },
						new IdentExpr { Name = "[lastname]" },
						new IdentExpr { Name = "[birth]" },
						new IdentExpr { Name = "[price]" }
					}
				},
				ValuesList = CreateSqlExprList(
					CreateSqlExprList(
						new IntegerExpr
						{
							Value = 267467
						},
						new StringExpr
						{
							Text = "N''"
						},
						new NullExpr
						{
							Token = "NULL"
						},
						new SqlFuncExpr
						{
							Name = "CAST",
							Arguments = new SqlExpr[]
							{
								new AsDataTypeExpr
								{
									Object = new Hex16NumberExpr { Value = "0x0000A5E5006236FB" },
									DataType = new DataTypeExpr { DataType = "DateTime" }
								}
							}
						},
						new IntegerExpr
						{
							Value = 1
						}
					)
				)
			}.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void insert_table_cast_float_as_decimal()
		{
			var sql = "INSERT [dbo].[customer] ([id]) VALUES ( CAST(0.0100 AS Decimal(5, 4)) )";
			var expr = Parse(sql);

			"INSERT [dbo].[customer]([id]) VALUES( CAST( 0.0100 AS Decimal(5,4) ) )".ShouldEqual(expr);
		}

		[Fact]
		public void insert_table_field_values_nstring_integer()
		{
			var sql = @"INSERT [dbo].[customer] ([id]) VALUES 
(N'127.0.0.1', 267467)";

			var expr = Parse(sql);
			new InsertExpr
			{
				Table = new IdentExpr
				{
					Name = "[customer]",
					ObjectId = "[dbo]"
				},
				Fields = CreateSqlExprList(
					new IdentExpr
					{
						Name = "[id]"
					}
				),
				ValuesList = CreateSqlExprList(
					CreateSqlExprList(
						new StringExpr
						{
							Text = "N'127.0.0.1'"
						},
						new IntegerExpr
						{
							Value = 267467
						}
					)
				)
			}.ToExpectedObject().ShouldEqual(expr);
		}

		[Fact]
		public void insert_into_variable_select_from()
		{
			var sql = @"insert into @table
		select Val
		from strsplitmax(@str, N',')";

			var expr = Parse(sql);

			"INSERT INTO @table SELECT Val FROM strsplitmax( @str,N',' )".ShouldEqual(expr);
		}

		[Fact]
		public void insert_into_table_columns_select_from()
		{
			var sql = @"insert into customer(id,name)
		select id, name from other";

			var expr = Parse(sql);

			"INSERT INTO customer SELECT id,name FROM other".ShouldEqual(expr);
		}

		[Fact]
		public void if_func_begin_insert_into_from_select_end()
		{
			var sql = @"if(isnull(@b1, '') <> '')
	begin
		insert into @a1
		select Val
		from
			strsplitmax(@str, N',')
	end";

			var expr = Parse(sql);
			@"IF (isnull( @b1,'' ) <> '')
BEGIN
INSERT INTO @a1 SELECT Val FROM strsplitmax( @str,N',' )
END".ShouldEqual(expr);
		}

		[Fact]
		public void insert_values_parentheses()
		{
			var sql = @"insert into @tb (a, b)
      values
        (
            (case when @c>=1 then 2 else 3 end), -- desc
            @d
        )";

			var expr = _sqlParser.ParseInsertPartial(sql);

			@"INSERT INTO @tb(a,b) VALUES( (CASE
WHEN @c >= 1 THEN 2
ELSE 3
END),@d )".ShouldEqual(expr);
		}

		[Fact]
		public void insert_into_variableName_values_()
		{
			var sql = @"insert into @a values(@d, @c)";

			var expr = _sqlParser.ParseInsertPartial(sql);

			@"INSERT INTO @a VALUES( @d,@c )".ShouldEqual(expr);
		}

		[Fact]
		public void insert_values()
		{
			var sql = @"INSERT (id, name) VALUES (source.id, source.name)";

			var expr = _sqlParser.Parse(sql);

			@"INSERT (id,name) VALUES(source.id,source.name)".ShouldEqual(expr);
		}

		[Fact]
		public void insert_into_table_output_inserted_into_table_columns()
		{
			var sql = @"insert into customer([id],[name]) 
				output 'customer', inserted.id, GETDATE()
				into customerTracker([id], [name])
				select id,name from otherCustomer";

			var expr = _sqlParser.ParseInsertPartial(sql);

			@"INSERT INTO customer 
OUTPUT 'customer',inserted.id,GETDATE() 
INTO customerTracker([id],[name])
SELECT id,name FROM otherCustomer".ShouldEqual(expr);
		}
	}
}