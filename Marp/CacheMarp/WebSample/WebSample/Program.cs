using AspectCore.Configuration;
using AspectCore.Extensions.DependencyInjection;
using WebSample.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddControllersWithViews();
builder.Services.AddTransient<IUserService, UserService>();
builder.Services.AddTransient<IGlobalSettingRepo, GlobalSettingRepo>();
builder.Services.AddTransient<IGlobalSettingService, GlobalSettingService>();
builder.Services.AddTransient<IGlobalSettingFactory<MyGlobalSettings>, GlobalSettingFactory<MyGlobalSettings>>();

//builder.Services.ConfigureDynamicProxy(config =>
//{
//	config.Interceptors.AddTyped<CacheInterceptorAttribute>();
//});


builder.Host.UseServiceProviderFactory(new DynamicProxyServiceProviderFactory());


var app = builder.Build();



// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
	app.UseExceptionHandler("/Home/Error");
}
app.UseStaticFiles();

app.UseRouting();

app.UseAuthorization();

app.MapControllerRoute(
	 name: "default",
	 pattern: "{controller=Home}/{action=Index}/{id?}");

app.Run();
