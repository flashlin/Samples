using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;

namespace VimSharpApp.ApiHandlers;

public class JobApiHandler : IJobApiHandler
{
    public Task<GetJobsInfoResponse> GetJobsInfo()
    {
        return Task.FromResult(new GetJobsInfoResponse
        {
            Jobs = [
                new JobInfo { Id = "1", Name = "Job1", Method = "Method1", Parameters = ["Param1", "Param2"] },
                new JobInfo { Id = "2", Name = "Job2", Method = "Method2", Parameters = ["Param3", "Param4"] }
            ]
        });
    }

    public static void MapEndpoints(WebApplication app)
    {
        app.MapPost("/api/GetJobsInfo", (IJobApiHandler handler) => handler.GetJobsInfo());
    }
}
