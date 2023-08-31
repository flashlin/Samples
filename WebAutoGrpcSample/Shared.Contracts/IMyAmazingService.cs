using System.ServiceModel;

namespace Shared.Contracts;

[ServiceContract]
//[Service("foo")]
public interface IMyAmazingService {
    
    [OperationContract]
    ValueTask<SearchResponse> SearchAsync(SearchRequest request);
}