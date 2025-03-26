using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;

namespace VimSharpApp.ApiHandlers;

public static class WelcomeHandler
{
    public static void MapWelcomeEndpoints(this WebApplication app)
    {
        app.MapGet("/SayHello", () => "Hi");
    }
} 