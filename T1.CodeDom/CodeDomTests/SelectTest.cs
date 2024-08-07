﻿using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests
{
    public class SelectTest : TestBase
    {
        public SelectTest(ITestOutputHelper outputHelper) : base(outputHelper)
        {
        }

        [Fact]
        public void select_number()
        {
            var sql = "select 1";
            Parse(sql);
            ThenExprShouldBe("SELECT 1");
        }
        
        
        [Fact]
        public void select_type()
        {
            var sql = "select type";
            Parse(sql);
            ThenExprShouldBe("SELECT type");
        }
        
        [Fact]
        public void select_matic_number()
        {
            var sql = "select 1/2, 2";
            Parse(sql);
            ThenExprShouldBe("SELECT 1 / 2, 2");
        }
        
        [Fact]
        public void select_nolock()
        {
            var sql = "select 1 from customer (nolock)";
            
            Parse(sql);
            
            ThenExprShouldBe("SELECT 1 FROM customer( nolock )");
        }
        
        [Fact]
        public void select_min()
        {
            var sql = "select min,name from customer";
            Parse(sql);
            ThenExprShouldBe("SELECT min, name FROM customer");
        }
        
        [Fact]
        public void select_top_10_percent()
        {
            var sql = "select top 10 percent name from customer";
            Parse(sql);
            ThenExprShouldBe("SELECT TOP 10 PERCENT name FROM customer");
        }
        

        [Fact]
        public void select_string_as_aliasName()
        {
            var sql = "select 'abc' as error";
            Parse(sql);
            ThenExprShouldBe("SELECT 'abc' AS error");
        }

        [Fact]
        public void select_number_comment_number_from_table()
        {
            var sql = @"select 1, --test
2 from customer";
            Parse(sql);

            ThenExprShouldBe(@"SELECT 1, 2 FROM customer");
        }

        [Fact]
        public void select_name()
        {
            var sql = "select name";
            Parse(sql);
            ThenExprShouldBe("SELECT name");
        }

        [Fact]
        public void select_date()
        {
            var sql = "select date from customer";
            Parse(sql);
            ThenExprShouldBe("SELECT date FROM customer");
        }
        
        [Fact]
        public void select_over()
        {
            var sql = "select date over(), ROW_NUMBER() over(order by name desc) from customer";
            Parse(sql);
            ThenExprShouldBe("SELECT date OVER( ), ROW_NUMBER() OVER( ORDER BY name DESC ) FROM customer");
        }
        

        [Fact]
        public void select_customFunc()
        {
            var sql = "select [dbo].[isMy](123)";
            Parse(sql);
            ThenExprShouldBe("SELECT [dbo].[isMy]( 123 )");
        }

        [Fact]
        public void select_field_customFunc()
        {
            var sql = "select name, [dbo].[isMy](123)";
            Parse(sql);
            ThenExprShouldBe("SELECT name, [dbo].[isMy]( 123 )");
        }


        [Fact]
        public void select_group_customFunc()
        {
            var sql = "select name, ([dbo].[isMy](123)) as a";
            Parse(sql);
            ThenExprShouldBe("SELECT name, ( [dbo].[isMy]( 123 ) ) AS a");
        }

        [Fact]
        public void select_where_eq_and_between()
        {
            var sql = @"select id from customer where 
id1=@id and
id2 between 1 and 38 ";
            Parse(sql);

            ThenExprShouldBe(@"SELECT id
FROM customer
WHERE id1 = @id AND id2 BETWEEN 1 AND 38");
        }

        [Fact]
        public void select_var_eq_var_from_table_where_column_in_var()
        {
            var sql = @"select @id = @id + 1	
from customer with(nolock)
where id in (@id1, @id2)";

            Parse(sql);

            ThenExprShouldBe(@"SELECT @id = @id + 1
FROM customer WITH(NOLOCK)
WHERE id IN (@id1, @id2)");
        }

        [Fact]
        public void select_name_from_table_order_by()
        {
            var sql = "select name from customer order by id";
            Parse(sql);
            ThenExprShouldBe(@"SELECT name
FROM customer
ORDER BY id ASC");
        }

        [Fact]
        public void select_name_from_table_group_by()
        {
            var sql = "select name from customer group by id";
            Parse(sql);
            ThenExprShouldBe(@"SELECT name
FROM customer
GROUP BY id");
        }


        [Fact]
        public void select_top_1_name()
        {
            var sql = "select top 1 name";
            Parse(sql);
            ThenExprShouldBe("SELECT TOP 1 name");
        }
        
        [Fact]
        public void select_top_arith()
        {
            var sql = "select top (@a-@b) name";
            Parse(sql);
            ThenExprShouldBe("SELECT TOP ( @a - @b ) name");
        }
        

        [Fact]
        public void select_name_number()
        {
            var sql = "select name, 1";
            Parse(sql);
            ThenExprShouldBe("SELECT name, 1");
        }

        [Fact]
        public void select_name_without_as_name()
        {
            var sql = "select customerName name";
            Parse(sql);
            ThenExprShouldBe("SELECT customerName AS name");
        }
        
        [Fact]
        public void select_with_nowait()
        {
            var sql = "select 1 from customer with(nowait)";
            Parse(sql);
            ThenExprShouldBe("SELECT 1 FROM customer WITH(NOWAIT)");
        }
        

        [Fact]
        public void select_name_as_name()
        {
            var sql = "select customerName as name";
            Parse(sql);
            ThenExprShouldBe("SELECT customerName AS name");
        }

        [Fact]
        public void select_number_from_dbo_table()
        {
            var sql = @"select 1 from dbo.customer";
            Parse(sql);
            ThenExprShouldBe("SELECT 1 FROM dbo.customer");
        }

        [Fact]
        public void select_number_from_table_aliasName()
        {
            var sql = @"select 1 from dbo.customer tb1";
            Parse(sql);
            ThenExprShouldBe("SELECT 1 FROM dbo.customer AS tb1");
        }

        [Fact]
        public void select_number_from_dbo_table_where()
        {
            var sql = @"select 1 from dbo.customer where name=customFunc()";
            Parse(sql);
            ThenExprShouldBe("SELECT 1 FROM dbo.customer WHERE name = customFunc()");
        }

        [Fact]
        public void select_number_from_dbo_table_where_and()
        {
            var sql = @"select 1 from dbo.customer where a=1 and b=2";
            Parse(sql);
            ThenExprShouldBe("SELECT 1 FROM dbo.customer WHERE a = 1 AND b = 2");
        }

        [Fact]
        public void select_number_union_all_select_number()
        {
            var sql = @"select 1 
union all
select 2";
            Parse(sql);
            ThenExprShouldBe("SELECT 1 UNION ALL SELECT 2");
        }

        [Fact]
        public void select_star_into_tmp()
        {
            var sql = @"select * into #tmp from customer";

            Parse(sql);

            ThenExprShouldBe(@"SELECT *
INTO #tmp
FROM customer");
        }

        [Fact]
        public void select_from_select()
        {
            var sql = @"SELECT c.id, p.*
        FROM (
                SELECT *
                FROM otherTable
                WHERE id = @id
        ) AS p
        JOIN customer c WITH (NOLOCK) ON p.id = c.id
        ORDER BY c.name, c.id;";

            Parse(sql);

            ThenExprShouldBe(@"SELECT c.id, p.*
FROM ( SELECT * FROM otherTable WHERE id = @id ) AS p
JOIN customer c WITH(NOLOCK) ON p.id = c.id
ORDER BY c.name ASC, c.id ASC ;");
        }


        [Fact]
        public void select_from_comment_select()
        {
            var sql = @"SELECT p.*
        FROM (
--test
                SELECT *
                FROM otherTable
                WHERE id = @id
        ) AS p
        ";

            Parse(sql);

            ThenExprShouldBe(@"SELECT p.*
FROM ( SELECT *
FROM otherTable
WHERE id = @id ) AS p");
        }


        [Fact]
        public void select_from_where_comment_comment()
        {
            var sql = @"select id
	from customer
	where c.id = 1 -- comment1
		and c.status & 2 <> 3 -- comment2
		and c.name is NULL
		";

            Parse(sql);

            ThenExprShouldBe(@"SELECT id
FROM customer
WHERE c.id = 1 AND c.status & 2 <> 3 AND c.name IS NULL");
        }


        [Fact]
        public void select_top_into_tmpTable_from_table()
        {
            var sql = @"SELECT TOP 1 * INTO #tmpCustomer from customer";

            Parse(sql);

            ThenExprShouldBe(@"SELECT TOP 1 *
INTO #tmpCustomer
FROM customer");
        }

        [Fact]
        public void select_with_index()
        {
            var sql = @"SELECT 1 from customer with(nolock, index(aaa))";

            Parse(sql);

            ThenExprShouldBe(@"SELECT 1
FROM customer WITH(NOLOCK, INDEX(AAA))");
        }

        [Fact]
        public void select_sum_as_out()
        {
            var sql = @"SELECT sum(id) as out, name from customer";

            Parse(sql);

            ThenExprShouldBe(@"SELECT SUM( id ) AS out, name FROM customer");
        }

        [Fact]
        public void select_name_as_count()
        {
            var sql = @"select 'main' as name, COUNT(1) Count";

            Parse(sql);

            ThenExprShouldBe(@"SELECT 'main' AS name, COUNT( 1 ) AS Count");
        }


        [Fact]
        public void select_having()
        {
            var sql = @"select 1 from customer where id=1 group by id having id > 1";

            Parse(sql);

            ThenExprShouldBe(@"SELECT 1 FROM customer
WHERE id = 1
GROUP BY id
HAVING id > 1");
        }


        [Fact]
        public void select_order_by()
        {
            var sql = @"select * from @customer c
	order by c.Status & 1, name";

            Parse(sql);

            ThenExprShouldBe(@"SELECT * FROM @customer AS c ORDER BY c.Status & 1 ASC, name ASC");
        }

        [Fact]
        public void select_for_xml()
        {
            var sql = @"select name from customer
for xml path('')              ";

            Parse(sql);

            ThenExprShouldBe(@"SELECT name FROM customer FOR XML PATH ( '' )");
        }

        [Fact]
        public void select_for_xml_auto()
        {
            var sql = @"select name from customer
       for xml auto, ROOT('customer')";

            Parse(sql);

            ThenExprShouldBe(@"SELECT name FROM customer FOR XML AUTO, ROOT ( 'customer' )");
        }

        [Fact]
        public void select_semecolon()
        {
            var sql = @"select name ;";

            Parse(sql);

            ThenExprShouldBe(@"SELECT name ;");
        }
        
        [Fact]
        public void select_string_type()
        {
            var sql = @"select 'aaa' type";

            Parse(sql);

            ThenExprShouldBe(@"SELECT 'aaa' AS type");
        }
        

        [Fact]
        public void select_where_between_fn()
        {
            var sql = @"select id from customer 
			where birth between @startDate and dateadd(hh,24,@startDate) and    
			id >= 100";

            Parse(sql);

            ThenExprShouldBe(@"SELECT id FROM customer
WHERE birth BETWEEN @startDate AND DATEADD( hh, 24, @startDate )
AND id >= 100");
        }
        
        
        [Fact]
        public void select_xml()
        {
            var sql = @"select id from @xml.nodes('/Array/Customer') AS XMLNode(Node)";

            Parse(sql);

            ThenExprShouldBe(@"SELECT id FROM @xml.NODES( '/Array/Customer' ) AS XMLNode(Node)");
        }
        
        
        [Fact]
        public void select_online()
        {
            var sql = @"select @id=id from Online with (nolock) where id=1";

            Parse(sql);

            ThenExprShouldBe(@"SELECT @id = id FROM Online WITH(NOLOCK) WHERE id = 1");
        }
        
        
        [Fact]
        public void select_var_eq_inserted()
        {
            var sql = @"select @id=inserted.id from inserted";

            Parse(sql);

            ThenExprShouldBe(@"SELECT @id = inserted.id FROM inserted");
        }
        
        
        [Fact]
        public void select_case_isnull()
        {
            var sql = @"select 
       case when id=1
			then id+2
		else id+3 end nid, 
		isnull(name, '123') name from customer";

            Parse(sql);

            ThenExprShouldBe(@"SELECT CASE WHEN id = 1 THEN id + 2 ELSE id + 3 END AS nid,
ISNULL( name, '123' ) AS name
FROM customer");
        }
        
        
        [Fact]
        public void select_sys_dm_exec_requests()
        {
            var sql = @"select der.command
from sys.dm_exec_requests der
    cross apply sys.dm_exec_sql_text(der.sql_handle) dest";

            Parse(sql);

            ThenExprShouldBe(@"SELECT der.command FROM sys.dm_exec_requests AS der
CROSS APPLY sys.dm_exec_sql_text( der.sql_handle ) AS dest");
        }
        
        
        [Fact]
        public void select_over_order_by_sum()
        {
            var sql = @"SELECT  ROW_NUMBER() OVER(ORDER BY Sum(BetCount) desc) AS ROWID, id from customer";

            Parse(sql);

            ThenExprShouldBe(@"SELECT ROW_NUMBER() OVER( ORDER BY SUM( BetCount ) DESC ) AS ROWID, id FROM customer");
        }
        
        
        [Fact]
        public void select_from_rent_inner_join()
        {
            var sql = @"SELECT id, (select name FROM [otherCustomer] WHERE [Id] = @id) AS [playerName]
		FROM 
		(
			[customer] WITH(NOLOCK) 
			INNER JOIN ( SELECT gid FROM sample ) AS [g] 
			ON [g].[gid] = 123
		) ORDER BY id";

            Parse(sql);

            ThenExprShouldBe(@"SELECT id, ( SELECT name FROM [otherCustomer] WHERE [Id] = @id ) AS [playerName]
FROM (
    [customer] WITH(NOLOCK) 
    INNER JOIN ( SELECT gid FROM sample ) [g] ON [g].[gid] = 123 
)
ORDER BY id ASC");
        }
        
        
    }
}