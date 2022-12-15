using System.ComponentModel.DataAnnotations.Schema;
using System.Diagnostics;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;
using MockApiWeb.Models;

namespace MockApiWeb.Controllers;

public class HomeController : Controller
{
    private readonly ILogger<HomeController> _logger;

    public HomeController(ILogger<HomeController> logger)
    {
        _logger = logger;
    }

    public IActionResult Index()
    {
        return View();
    }

    public IActionResult Privacy()
    {
        return View();
    }

    [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
    public IActionResult Error()
    {
        return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
    }
}

public class MockDbContext : DbContext
{
    public MockDbContext(DbContextOptions<MockDbContext> options)
        : base(options)
    {
    }

    public DbSet<WebApiFuncInfoEntity> WebApiFuncInfos { get; set; } = null!;
}

[Table("WebApiFuncInfos")]
public class WebApiFuncInfoEntity
{
    [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
    public int Id { get; set; }
    public string ProductName { get; set; } = string.Empty;
    public string ActionName { get; set; } = string.Empty;
    public WebApiAccessMethodType Method { get; set; }
    public string ResponseContent { get; set; } = string.Empty;
}

public enum WebApiAccessMethodType
{
    Post,
    Get
}

public class DbContextFactory
{
    public TContext Create<TContext>()
        where TContext: DbContext
    {
        var cn = new SqliteConnection("data source=:memory:");
        cn.Open();
        var opt = new DbContextOptionsBuilder<TContext>()
            .UseSqlite(cn).Options;
        var db = (TContext)Activator.CreateInstance(typeof(TContext), opt)!;
        db.Database.EnsureCreated();
        return db;
    }
}