namespace WebAutoGrpcSample.GrpcServices;

public class MyServer : IMyAmazingService {
    public ValueTask<SearchResponse> SearchAsync(SearchRequest request)
    {
        throw new NotImplementedException();
    }
}