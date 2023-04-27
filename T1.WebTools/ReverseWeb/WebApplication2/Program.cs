using System.Diagnostics;
using System.Net;
using Yarp.ReverseProxy.Forwarder;
using Yarp.ReverseProxy.Utilities;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddControllersWithViews();
builder.Services.AddHttpForwarder();

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

void ConfigureMap(IEndpointRouteBuilder endpointRouteBuilder)
{
    var httpClient = new HttpMessageInvoker(new SocketsHttpHandler()
    {
        UseProxy = false,
        AllowAutoRedirect = false,
        AutomaticDecompression = DecompressionMethods.None,
        UseCookies = false,
        ActivityHeadersPropagator = new ReverseProxyPropagator(DistributedContextPropagator.Current),
        ConnectTimeout = TimeSpan.FromSeconds(15),
    });
    var transformer = HttpTransformer.Default;
    var requestConfig = new ForwarderRequestConfig { ActivityTimeout = TimeSpan.FromSeconds(100) };
    var forwarder = endpointRouteBuilder.ServiceProvider.GetRequiredService<IHttpForwarder>();
    endpointRouteBuilder.Map("/App1/{**catch-all}", async httpContext =>
    {
        var error = await forwarder.SendAsync(httpContext, "http://localhost:5000/",
            httpClient, requestConfig, transformer);
        if (error != ForwarderError.None)
        {
            var errorFeature = httpContext.GetForwarderErrorFeature();
            var exception = errorFeature.Exception;
        }
    });
}

app.UseEndpoints(ConfigureMap);


app.UseAuthorization();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}");

app.Run();