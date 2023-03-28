﻿using System.ServiceModel;

namespace Shared.Contracts;

[ServiceContract]
//[Service("foo")]
public interface IMyAmazingService {
    ValueTask<SearchResponse> SearchAsync(SearchRequest request);
}