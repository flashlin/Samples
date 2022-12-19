using System.Runtime.Serialization;

namespace MockApiWeb.Models.DataConstraints;

[DataContract]
public class QueryWebApiSimpleSettingForm
{
    [DataMember(Name = "ProductName")] public string ProductName { get; set; } = string.Empty;
    [DataMember(Name = "ControllerName")] public string ControllerName { get; set; } = string.Empty;
    [DataMember(Name = "ActionName")] public string ActionName { get; set; } = string.Empty;
    [DataMember(Name = "ResponseContent")] public string ResponseContent { get; set; } = string.Empty;
}