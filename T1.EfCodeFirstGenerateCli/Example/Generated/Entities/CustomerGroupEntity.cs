using System;

namespace Generated
{
    public class CustomerGroupEntity
    {
        public int GroupID { get; set; }
        public int PGroupID { get; set; }
        public required string GroupName { get; set; }
        public byte GroupLevel { get; set; }
        public int Status { get; set; }
        public decimal RiskLimit { get; set; }
        public DateTime CreatedOn { get; set; }
        public required string CreatedBy { get; set; }
        public DateTime? ModifiedOn { get; set; }
        public required string ModifiedBy { get; set; }
        public required string Auditor { get; set; }
        public DateTime? LastAuditOn { get; set; }
        public required string LastAuditBy { get; set; }
        public int? ExtraInfoID { get; set; }
        public required string Remark { get; set; }
        public decimal? SuspendLimit { get; set; }
        public bool? GroupSuspendStatus { get; set; }
        public bool? GroupSuspendIsEnabled { get; set; }
        public DateTime? KYCExpiryDate { get; set; }
    }
}
