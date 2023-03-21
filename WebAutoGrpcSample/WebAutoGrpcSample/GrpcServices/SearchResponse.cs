using System.Runtime.Serialization;

namespace WebAutoGrpcSample.GrpcServices;

public class SearchResponse
{
    [DataMember(Order = 1)]
    public int Id { get; set; }
}