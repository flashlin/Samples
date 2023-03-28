using System.Runtime.Serialization;

namespace Shared.Contracts;

[DataContract]
public class SearchRequest
{
    [DataMember(Order = 1)]
    public string Name { get; set; }
}