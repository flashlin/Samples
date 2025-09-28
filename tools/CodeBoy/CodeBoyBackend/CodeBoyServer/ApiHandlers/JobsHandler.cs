using MassTransit;
using CodeBoyLib.MQ;
using Microsoft.AspNetCore.Mvc;

namespace CodeBoyServer.ApiHandlers
{
    /// <summary>
    /// Handler for job management API endpoints
    /// </summary>
    public static class JobsHandler
    {
        /// <summary>
        /// Configure job management endpoints
        /// </summary>
        /// <param name="app">Web application builder</param>
        public static void Map(WebApplication app)
        {
            app.MapPost("/api/jobs/create", CreateJob)
                .WithName("CreateJob")
                .WithDescription("Create a new job and return job ID")
                .WithTags("Jobs")
                .WithOpenApi();
            
            app.MapGet("/api/jobs/getProgress", GetProgress)
                .WithName("GetProgress")
                .WithDescription("Get job progress via Server-Sent Events")
                .WithTags("Jobs")
                .WithOpenApi();
        }

        /// <summary>
        /// Create a new job endpoint handler
        /// </summary>
        /// <param name="publisher">MassTransit publish endpoint</param>
        /// <returns>Job ID</returns>
        private static async Task<string> CreateJob(IPublishEndpoint publisher)
        {
            var jobId = Guid.NewGuid().ToString();
            await publisher.Publish<IStartJob>(new StartJobMessage
            {
                JobId = jobId
            });
            return jobId;
        }

        /// <summary>
        /// Get job progress via Server-Sent Events endpoint handler
        /// </summary>
        /// <param name="jobId">Job ID to monitor</param>
        /// <param name="progressQueue">Progress queue service</param>
        /// <param name="response">HTTP response</param>
        /// <param name="ct">Cancellation token</param>
        private static async Task GetProgress(
            string jobId,
            [FromServices] IProgressQueue progressQueue,
            HttpResponse response,
            CancellationToken ct)
        {
            response.ContentType = "text/event-stream";
            response.Headers.Add("Cache-Control", "no-cache");
            response.Headers.Add("Connection", "keep-alive");
            
            await foreach (var message in progressQueue.Consume(jobId, ct))
            {
                await response.WriteAsync($"data: {message}\n\n", ct);
                await response.Body.FlushAsync(ct);
            }
        }
    }

    /// <summary>
    /// Implementation of IStartJob for message publishing
    /// </summary>
    public class StartJobMessage : IStartJob
    {
        public string JobId { get; set; } = string.Empty;

        public async Task Execute()
        {
            // Simulate some work
            await Task.Delay(1000);
        }
    }
}