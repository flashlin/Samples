using System;

namespace Generated
{
    public class AuditGroupMoveLogEntity
    {
        public DateTime MoveDate { get; set; }
        public int ChildAuditGroupId { get; set; }
        public int? IomCustomerMappingCount { get; set; }
        public int? SboCustomerMappingCount { get; set; }
        public int? PreviousParentAuditGroupId { get; set; }
        public int NewParentAuditGroupId { get; set; }
        public required string CreatedBy { get; set; }
    }
}
