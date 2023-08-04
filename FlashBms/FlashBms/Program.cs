using FlashBms.Models.Repositories;
using Microsoft.OpenApi.Models;
using T1.AspNetCore.Extensions;

var builder = WebApplication.CreateBuilder(args);
var services = builder.Services;
services.AddTransient<BannerDbContext>();
services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new OpenApiInfo { Title = "Banner Mgmt API", Version = "v1" });
});
services.AddControllersWithViews();
services.AddViewToStringRendererService();

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

app.UseSwagger();
app.UseSwaggerUI(c =>
{
    c.SwaggerEndpoint("/swagger/v1/swagger.json", "Your API v1");
});


app.Run();