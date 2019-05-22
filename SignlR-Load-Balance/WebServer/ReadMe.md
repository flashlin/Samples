Startup.cs
```
public void ConfigureServices(IServiceCollection services)
{
	services.AddSignalR()
}
```

```
public void Configure(IApplicationBuilder app, IHostingEnvironment env)
{
	app.UseSignalR(routes =>
	{
		routes.MapHub<ChatHub>("/hub");
	});
}
```