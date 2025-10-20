using System;

namespace Generated
{
    public class SensitiveInfoPermissionRequestEntity
    {
        public int Id { get; set; }
        public int? ApproverCustomerId { get; set; }
        public int RequesterCustomerId { get; set; }
        public required string RequesterAccountId { get; set; }
        public int RequestedCustomerId { get; set; }
        public required string RequestedAccountId { get; set; }
        public required string PageKey { get; set; }
        public required string Reason { get; set; }
        public DateTime RequestedOn { get; set; }
        public DateTime? ApprovedOn { get; set; }
        public DateTime? UsedOn { get; set; }
        public required string Department { get; set; }
    }
}
