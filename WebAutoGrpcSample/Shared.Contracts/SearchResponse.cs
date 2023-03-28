using System.Runtime.Serialization;

namespace Shared.Contracts;

[DataContract]
public class SearchResponse
{
    [DataMember(Order = 1)]
    public int Id { get; set; }
}