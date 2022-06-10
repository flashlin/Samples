using Microsoft.EntityFrameworkCore;
using PizzaWeb.Models;
using PizzaWeb.Models.Banner;
using PizzaWeb.Models.Helpers;
using PizzaWeb.Models.Repos;
using T1.AspNetCore.Extensions;
using T1.AspNetCore.FileProviders.Virtual;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddControllersWithViews()
		.AddRazorRuntimeCompilation(options =>
		{
			var dbConfigSection = builder.Configuration.GetSection("DbConfig");
			var connectionString = dbConfigSection["ConnectionString"];
			var dynamicFileProvider = new DynamicBannerTemplateFileProvider(connectionString);
			options.FileProviders.Add(new DynamicVirtualFileProvider(dynamicFileProvider));
		});
builder.Services.AddViewToStringRendererService();












builder.Services.AddTransient<IJsonConverter, JsonConverter>();
builder.Services.Configure<PizzaDbConfig>(builder.Configuration.GetSection("DbConfig"));
builder.Services.AddTransient<IDbContextOptionsFactory, SqlServerDbContextOptionsFactory>();
builder.Services.AddTransient<IPizzaRepo, PizzaRepo>();
//builder.Services.AddDbContext<PizzaDbContext>();
//builder.Services.AddDbContextFactory<PizzaDbContext>();
builder.Services.AddDbContextPool<PizzaDbContext>(o => 
	o.UseSqlServer(builder.Configuration
		.GetSection("DbConfig")
		.GetValue<string>("ConnectionString")));
//builder.Services.AddSingleton<IRepositoryFactory, RepositoryFactory>();
//builder.Services.AddTransient(sp => ActivatorUtilities.CreateInstance<PizzaDbContext>(sp));


var MyAllowSpecificOrigins = "_myAllowSpecificOrigins";
builder.Services.AddCors(options =>
{
	options.AddPolicy(name: MyAllowSpecificOrigins,
		policy =>
		{
			policy.WithOrigins("http://localhost:3000")
				.AllowAnyHeader()
				.AllowAnyMethod();
		});
});

var app = builder.Build();
ServiceLocator.SetLocatorProvider(app.Services);

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
	app.UseExceptionHandler("/Home/Error");
}
app.UseStaticFiles();

app.UseRouting();
app.UseCors(MyAllowSpecificOrigins);

app.UseAuthorization();

app.MapControllerRoute(
	 name: "default",
	 pattern: "{controller=Home}/{action=Index}/{id?}");

app.Run();
