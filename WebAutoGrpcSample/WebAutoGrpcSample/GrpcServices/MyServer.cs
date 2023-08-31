using Shared.Contracts;

namespace WebAutoGrpcSample.GrpcServices;

/**
 * 為了實現這一點，使用了以下 NuGet 包：
Grpc.Net.Client (used on Client project)
Grpc.Net.Client （用於客戶端項目）
protobuf-net (used on Domain project)
protobuf-net （用於域項目）
protobuf-net.Grpc (used on Client and Domain project)
protobuf-net.Grpc （用於客戶端和域項目）
protobuf-net.Grpc.AspNetCore (used on Service project)
protobuf-net.Grpc.AspNetCore （用於服務項目）
 */

public class MyServer : IMyAmazingService {
    public ValueTask<SearchResponse> SearchAsync(SearchRequest request)
    {
        return ValueTask.FromResult(new SearchResponse
        {
            Id = 1234
        });
    }
}