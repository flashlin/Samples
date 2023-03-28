using Shared.Contracts;

namespace WebAutoGrpcSample.GrpcServices;

public class MyServer : IMyAmazingService {
    public ValueTask<SearchResponse> SearchAsync(SearchRequest request)
    {
        return ValueTask.FromResult(new SearchResponse
        {
            Id = 123
        });
    }
}