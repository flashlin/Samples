using System.ServiceModel;
using ProtoBuf.Grpc.Configuration;

namespace WebAutoGrpcSample.GrpcServices;

[ServiceContract]
[Service("foo")]
public interface IMyAmazingService {
    ValueTask<SearchResponse> SearchAsync(SearchRequest request);
}