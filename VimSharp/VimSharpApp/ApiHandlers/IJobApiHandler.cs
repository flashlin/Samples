namespace VimSharpApp.ApiHandlers;

public interface IJobApiHandler
{
    Task<GetJobsInfoResponse> GetJobsInfo();
}