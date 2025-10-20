using System;

namespace Generated
{
    public class AuditCheckedListEntity
    {
        public int rid { get; set; }
        public int custid { get; set; }
        public required string auditor { get; set; }
        public decimal? cashbalance { get; set; }
        public decimal? totalbalance { get; set; }
        public DateTime auditDate { get; set; }
        public required string remark { get; set; }
        public DateTime createdDate { get; set; }
        public required string unCheckAudit { get; set; }
    }
}
