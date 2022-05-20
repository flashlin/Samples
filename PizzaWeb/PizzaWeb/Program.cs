using PizzaWeb.Models;
using T1.AspNetCore.Extensions;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddControllersWithViews();
builder.Services.Configure<PizzaDbConfig>(builder.Configuration.GetSection("DbConfig"));
builder.Services.AddTransient<IDbContextOptionsFactory, SqlServerDbContextOptionsFactory>();
//builder.Services.AddDbContext<PizzaDbContext>();
builder.Services.AddTransient(sp => ActivatorUtilities.CreateInstance<PizzaDbContext>(sp));
builder.Services.AddViewToStringRendererService();

var  MyAllowSpecificOrigins = "_myAllowSpecificOrigins";
builder.Services.AddCors(options =>
{
	options.AddPolicy(name: MyAllowSpecificOrigins,
		policy  =>
		{
			policy.WithOrigins("http://localhost:3000")
				.AllowAnyHeader()
				.AllowAnyMethod();
		});
});

var app = builder.Build();

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
