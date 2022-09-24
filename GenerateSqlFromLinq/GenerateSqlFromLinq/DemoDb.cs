using System.Collections.Concurrent;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Query.Internal;
using Microsoft.EntityFrameworkCore.Storage;

public class DemoDbContext : DbContext
{
	//public DemoDbContext(DbContextOptions<DemoDbContext> options) : base(options)
	//{
	//}

	public DbSet<Customer> Customers { get; set; }
	public DbSet<House> Houses { get; set; }
	public ConcurrentQueue<string> SqlLog { get; set; } = new();

	public static void Initialize(DemoDbContext context)
	{
		//if (!context.Database.EnsureCreated()) return;

		context.Database.OpenConnection();

		using var command = context.Database.GetDbConnection().CreateCommand();
		command.CommandText = @"
IF NOT EXISTS (SELECT TOP 1 1 FROM sys.objects WHERE object_id=OBJECT_ID(N'[dbo].[Customer]') AND type in (N'U'))
BEGIN
CREATE TABLE Customer
(
	Id int IDENTITY(1,1) NOT NULL PRIMARY KEY,
	Name nvarchar(50),
)
END


IF NOT EXISTS (SELECT TOP 1 1 FROM sys.objects WHERE object_id=OBJECT_ID(N'[dbo].[House]') AND type in (N'U'))
BEGIN
CREATE TABLE House
(
	Id int IDENTITY(1,1) NOT NULL PRIMARY KEY,
	Address nvarchar(100),
	BuyTime DATETIME,
   CustomerId int,
)
END
";
		command.ExecuteNonQuery();
	}


	public void WriteLog(Action action,
		[CallerMemberName] string memberName = "",
		[CallerFilePath] string filePath = "",
		[CallerLineNumber] int lineNumber = -1)
	{
		action();
		var log = GetLog();
		var message = $"{filePath} {lineNumber} {memberName}\r\n{log}";
		Console.WriteLine(message);
	}

	public string GetLog()
	{
		var sb = new StringBuilder();
		if (SqlLog.TryDequeue(out var line))
		{
			sb.AppendLine(line);
		}
		return sb.ToString();
	}

	protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
	{
		if (!optionsBuilder.IsConfigured)
		{
			var connectionString =
				@"server=localhost\SQLEXPRESS;database=North;trusted_connection=true;Integrated Security=True;";
			//optionsBuilder.UseInMemoryDatabase("DemoDb");
			optionsBuilder.UseSqlServer(connectionString);
			//localhost\SQLEXPRESS
		}

		optionsBuilder.AddInterceptors(new LogCommandInterceptor(SqlLog));
		//optionsBuilder.LogTo(Console.WriteLine)
		//	.LogTo(s => SqlLog.Enqueue(s));
	}
}


//public static class IQueryableExtensions
//{
//	private static readonly TypeInfo QueryCompilerTypeInfo = typeof(QueryCompiler).GetTypeInfo();

//	private static readonly FieldInfo QueryCompilerField = typeof(EntityQueryProvider).GetTypeInfo().DeclaredFields.First(x => x.Name == "_queryCompiler");

//	private static readonly FieldInfo QueryModelGeneratorField = QueryCompilerTypeInfo.DeclaredFields.First(x => x.Name == "_queryModelGenerator");

//	private static readonly FieldInfo DataBaseField = QueryCompilerTypeInfo.DeclaredFields.Single(x => x.Name == "_database");

//	private static readonly PropertyInfo DatabaseDependenciesField = typeof(Database).GetTypeInfo().DeclaredProperties.Single(x => x.Name == "Dependencies");

//	public static string ToSql<TEntity>(this IQueryable<TEntity> query) where TEntity : class
//	{
//		var queryCompiler = (QueryCompiler)QueryCompilerField.GetValue(query.Provider);
//		var modelGenerator = (QueryModelGenerator)QueryModelGeneratorField.GetValue(queryCompiler);
//		var queryModel = modelGenerator.ParseQuery(query.Expression);
//		var database = (IDatabase)DataBaseField.GetValue(queryCompiler);
//		var databaseDependencies = (DatabaseDependencies)DatabaseDependenciesField.GetValue(database);
//		var queryCompilationContext = databaseDependencies.QueryCompilationContextFactory.Create(false);
//		var modelVisitor = (RelationalQueryModelVisitor)queryCompilationContext.CreateQueryModelVisitor();
//		modelVisitor.CreateQueryExecutor<TEntity>(queryModel);
//		var sql = modelVisitor.Queries.First().ToString();

//		return sql;
//	}
//}