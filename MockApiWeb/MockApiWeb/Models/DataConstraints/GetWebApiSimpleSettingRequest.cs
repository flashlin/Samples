using System.Runtime.Serialization;

namespace MockApiWeb.Models.DataConstraints;

[DataContract]
public class GetWebApiSimpleSettingRequest
{
    [DataMember]
    public string ProductName { get; set; } = string.Empty;
    [DataMember]
    public string ControllerName { get; set; } = string.Empty;
    [DataMember]
    public string ActionName { get; set; } = string.Empty;
    [DataMember]
    public string ResponseContent { get; set; } = string.Empty;

    [DataMember] 
    public int StartId { get; set; }

    [DataMember] public int PageSize { get; set; } = 20;
}