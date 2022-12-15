using MockApiWeb.Controllers;
using MockApiWeb.Models.Middlewares;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddControllersWithViews();

var services = builder.Services;
services.AddSingleton<DynamicApiTransformer>();
services.AddSingleton<DbContextFactory>();
services.AddSingleton<MockDbContext>(sp =>
{
    var fac = sp.GetRequiredService<DbContextFactory>();
    return fac.Create<MockDbContext>();
});


var app = builder.Build();

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

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}");

app.MapDynamicControllerRoute<DynamicApiTransformer>("mock_{product}/api/{controller}/{action}");

app.Run();