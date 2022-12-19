using System.Runtime.Serialization;

namespace MockApiWeb.Models.DataConstraints;

[DataContract]
public class GetWebApiSimpleSettingRequest
{
    [DataMember(Name = "QueryForm")] public QueryWebApiSimpleSettingForm QueryForm { get; set; } = new();
    [DataMember(Name = "StartId")] public int StartId { get; set; }
    [DataMember(Name = "PageSize")] public int PageSize { get; set; } = 20;
}