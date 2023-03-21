using System.Runtime.Serialization;

namespace WebAutoGrpcSample.GrpcServices;

[DataContract]
public class SearchRequest
{
    [DataMember(Order = 1)]
    public string Name { get; set; }
}