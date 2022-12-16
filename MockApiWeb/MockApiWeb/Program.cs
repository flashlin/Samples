using Microsoft.AspNetCore.Mvc.ModelBinding;
using MockApiWeb.Controllers;
using MockApiWeb.Controllers.Apis;
using MockApiWeb.Models.Binders;
using MockApiWeb.Models.Middlewares;
using MockApiWeb.Models.Repos;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
var services = builder.Services;
services.AddControllersWithViews();
services.AddTransient<MockWebApiRequestBinder>();
services.AddControllers(options =>
{
    options.ModelBinderProviders.Insert(0, new MockWebApiBinderProvider());
});
services.AddSingleton<DynamicApiTransformer>();
services.AddSingleton<DbContextFactory>();
services.AddSingleton(sp =>
{
    var factory = sp.GetRequiredService<DbContextFactory>();
    return factory.Create<MockDbContext>();
});
services.AddSingleton<IMockDbRepo, MockDbRepo>();
services.AddCors();
services.AddSwaggerGen();


var app = builder.Build();
SQLitePCL.raw.SetProvider(new SQLitePCL.SQLite3Provider_e_sqlite3());


// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();

app.UseRouting();
app.UseAuthorization();

app.UseCors(options => options.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader());
app.UseSwagger();
app.UseSwaggerUI();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}");

app.MapDynamicControllerRoute<DynamicApiTransformer>("mock_{product}/api/{controller}/{action}");

app.Run();