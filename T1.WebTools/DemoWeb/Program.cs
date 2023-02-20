using T1.WebTools;
using T1.WebTools.Controllers;
using T1.WebTools.CsvEx;

var builder = WebApplication.CreateBuilder(args);


// Add services to the container.
builder.Services.AddControllersWithViews();
builder.Services.AddWebTools();

var services = builder.Services;
services.AddCors(options =>
{
    options.AddPolicy("AllowAll",
        cp => cp.AllowAnyOrigin()
            .AllowAnyMethod()
            .AllowAnyHeader());
});
// services.AddRazorPages()
//     .AddRazorPagesOptions(options => { options.AddTagHelperAssembly(typeof(CsvHeadersTypeSelectTagHelper).Assembly); });


var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
}

app.UseCors("AllowAll");
app.UseWebTools();
app.UseStaticFiles();

app.UseRouting();

app.UseAuthorization();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}");


app.Run();